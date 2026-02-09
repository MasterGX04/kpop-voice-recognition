import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog, colorchooser
import os
import json
from PIL import Image, ImageTk
from pathlib import Path
from audio_processing import combineMemberVocals
from group_registry import GroupRegistry
import sys
from urllib.parse import urlparse, urlunparse, quote
import urllib.request
import io, shutil
from audio_tester import VoiceDetectionApp
from util_functions import findBestAudioFiles, pickBestAudioForStem, ModalGuard
from image_generator import make_member_card, make_dark_member_card

class VoiceTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.groupRegistry = GroupRegistry(
            iconsRoot="group_icons",
            groupsJsonPath = "groups.json",
        )
        self.groups = self.groupRegistry.groups
        groupNames = list(self.groups.keys())
        self.currentGroup = tk.StringVar(value=groupNames[0] if groupNames else "")
        self.imageSize = (100, 100)
        self.groupMenu = None
        self.cacheMenu = None
        self.menuBar = None
        self.memberPhotos = {}
        self.memberImageRefs = []
        self.initPlaceholders()
        
        self.root.title("Voice Trainer")
        self.setResponsiveGeometry(self.root)
        self.generateImagesMenuIndex = None
        self.setMediaFolderMenuIndex = None
        
        self.createMenuBar()
        self.createWidgets()
        
        if self.currentGroup.get():
            self.displayMembers(self.currentGroup.get())
        else:
            self.showEmptyGroupsState()
            
        self.updateGroupMenuState()
    
    def createMenuBar(self):
        menubar = tk.Menu(self.root)
        self.menubar = menubar

        self.groupMenu = tk.Menu(menubar, tearoff=0)
        self.groupMenu.add_command(label="Add New Group...", command=self.openAddGroupDialog)
        
        # --- Edit Current Group (initially disabled) ---
        self.groupMenu.add_command(
            label="Edit Current Group...",
            command = self.openEditGroupDialog,
            state="disabled"
        )
        
        self.groupMenu.add_command(
            label="Set Media Folder...",
            command=self.openSetMediaFolderDialog,
            state="disabled"
        )
        self.setMediaFolderMenuIndex = self.groupMenu.index("end")
        
        self.groupMenu.add_command(
            label="Generate Bright/Dark Member Images...",
            command=self.openGenerateMemberImagesDialog,
            state="disabled"
        )
        self.generateImagesMenuIndex = self.groupMenu.index("end")
        
        self.groupMenu.add_separator()
        self.groupMenu.add_command(label="Rescan Groups", command=self.rescanGroups)
        
        menubar.add_cascade(label="Groups", menu=self.groupMenu)
        
        # --- Cache Menu ---
        self.cacheMenu = tk.Menu(menubar, tearoff=0, postcommand=self.refreshCacheMenuLabel)
        self.cacheMenu.add_command(label="Clear cache: ...", command=self.onClearCache)

        menubar.add_cascade(label="Cache", menu=self.cacheMenu)

        self.root.config(menu=menubar)
        self.root.config(menu=menubar)
    
    def updateGroupMenuState(self):
        """
        Enable or disable 'Edit Current Group...' based on whether a group is selected.
        """
        hasGroup = bool(self.currentGroup.get())
        
        # index 1 corresponds to "Edit Current Group"
        self.groupMenu.entryconfig(
            1,
            state="normal" if hasGroup else "disabled"
        )
        
        if self.generateImagesMenuIndex is not None:
            self.groupMenu.entryconfig(
                self.generateImagesMenuIndex,
                state="normal" if hasGroup else "disabled"
            )
            
        if self.setMediaFolderMenuIndex is not None:
            self.groupMenu.entryconfig(
                self.setMediaFolderMenuIndex,
                state="normal" if hasGroup else "disabled"
            )
        
    def getActiveDialogParent(self):
        # Prefer song picker if it exists; otherwise fall back to root
        try:
            if hasattr(self, "songPickerWindow") and self.songPickerWindow and self.songPickerWindow.winfo_exists():
                return self.songPickerWindow
        except Exception:
            pass
        return self.root
      
    def setResponsiveGeometry(self, window, scale=0.8, aspect=1920/1080):
        """Set window size based on monitor, keeping ~1920x1080 aspect ratio."""
        window.update_idletasks()
        
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        # Leave some margin to not touch scren edges
        max_w = int(screen_w * scale)   # e.g., 90% of screen
        max_h = int(screen_h * scale)
        
        # print(f"Max_w: {max_w}, max_h: {max_h}")
        
        # Start by using max height and compute width from aspect ratio
        width_from_h = int(max_h * aspect)
        
        if width_from_h <= max_w:
            # Height is the limiting factor
            win_w = width_from_h
            win_h = max_h
        else:
            # Width is the limiting factor; compute height from width
            win_w = max_w
            win_h = int(max_w / aspect)
            
        # Center window on the screen
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
           
    def createWidgets(self):
        # Topframe: group select
        topFrame = tk.Frame(self.root)
        topFrame.pack(pady=10)
        
        groupLabel = tk.Label(topFrame, text="Choose a K-pop group:")
        groupLabel.pack(side=tk.LEFT, padx=5)
        
        self.groupDropdown = ttk.Combobox(
            topFrame,
            textvariable=self.currentGroup,
            values=list(self.groups.keys()),
            state="readonly"
        )
        self.groupDropdown.pack(side=tk.LEFT)
        self.groupDropdown.bind(
            "<<ComboboxSelected>>",
            lambda e: (self.displayMembers(self.currentGroup.get()), self.updateGroupMenuState())
        )
        
        # Center: Scrollable vertical list
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            container,
            highlightthickness=0,
            bd=0,
            relief="flat"
        )
        self.scrollFrame = tk.Frame(self.canvas)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.scrollFrame, anchor="nw")
        
        self.scrollFrame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.enableGlobalScroll(self.canvas, container)
        
        # Button frame
        bottomFrame = tk.Frame(self.root)
        bottomFrame.pack(pady=10)
        
        #tk.Button(bottomFrame, text="Train", command=self.trainModel).pack(side=tk.LEFT, padx=10)
        tk.Button(bottomFrame, text="Select Song", command=self.selectSong).pack(side=tk.LEFT, padx=10)
        # tk.Button(bottomFrame, text="Extract Member Vocals", command=self.combineAllVocalsFromGroup).pack(side=tk.LEFT, padx=10)
        
        self.statusVar = tk.StringVar(
            value="Tip: right-click a singer’s image to set a custom image URL (.png/.jpg)."
        )
        statusFrame = tk.Frame(self.root)
        statusFrame.pack(fill="x", padx=10, pady=(0, 8))

        self.statusLabel = tk.Label(
            statusFrame,
            textvariable=self.statusVar,
            anchor="w",
            fg="gray"
        )
        self.statusLabel.pack(fill="x")

    def showEmptyGroupsState(self):
        for widget in self.scrollFrame.winfo_children():
            widget.destroy()

        self.memberImageRefs.clear()

        frame = tk.Frame(self.scrollFrame, pady=20)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="No groups found.\nUse Groups → Add New Group… to create one.",
            font=("Helvetica", 16),
            fg="gray"
        ).pack(padx=20, pady=20)
    
    def rescanGroups(self):
        self.groupRegistry.scanForNewGroups()
        self.groupRegistry.fillMissingColorsFromIcons()
        self.groupRegistry.saveGroupsToJson()
        
        self.groups = self.groupRegistry.groups
        groupNames = list(self.groups.keys())
        
        if hasattr(self, "groupDropdown") and self.groupDropdown:
            self.groupDropdown["values"] = groupNames

        if not groupNames:
            self.currentGroup.set("")
            self.showEmptyGroupsState()
            return

        if self.currentGroup.get() not in self.groups:
            self.currentGroup.set(groupNames[0])

        self.displayMembers(self.currentGroup.get())
        self.updateGroupMenuState()
    
    def displayMembers(self, groupName):
        if not groupName or groupName not in self.groups:
            self.showEmptyGroupsState()
            return
    
        for widget in self.scrollFrame.winfo_children():
            widget.destroy()
            
        self.memberImageRefs.clear()
        
        members = self.groups[groupName]["members"]
        
        for member in members:
            memberName = member['name']
            frame = tk.Frame(self.scrollFrame, pady=5)
            frame.pack(fill="x", padx=0)

            image = self.loadMemberImage(groupName, member)
            self.memberImageRefs.append(image)
            
            labelImage = tk.Label(frame, image=image)
            labelImage.pack(side="left")
            
            labelImage.bind(
                "<Button-3>",
                lambda e, g=groupName, m=member: self.openMemberImageMenu(e, g, m)
            )

            labelText = tk.Label(frame, text=memberName, font=("Helvetica", 18), fg="black")
            labelText.pack(side="left", padx=20)
    
    def openEditGroupDialog(self):
        groupName = self.currentGroup.get()
        if not groupName:
            return  # safety guard, menu should already be disabled

        groupDir = self.groupRegistry.iconsRoot / groupName
        manifest = self.groupRegistry._loadGroupManifest(groupDir)
        if not manifest:
            messagebox.showerror(
            "Edit Group",
            f"No group.json found for '{groupName}'.\n\n"
            "Use Groups → Add New Group… to create a manifest first."
            )
            return
        
        groupDir = self.groupRegistry.getGroupDir(groupName)
        
        win = tk.Toplevel(self.root)
        win.title(f"Edit Group: {groupName}")
        win.transient(self.root)
        win.grab_set()

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)
        
        # Theme dropdown
        ttk.Label(frm, text="Theme subfolder:").grid(row=0, column=0, sticky="w")
        themeVar = tk.StringVar(value=manifest.get("activeTheme", ""))

        subfolders = sorted([p.name for p in groupDir.iterdir() if p.is_dir()], key=str.lower)
        themeDropdown = ttk.Combobox(frm, textvariable=themeVar, values=subfolders, state="readonly")
        themeDropdown.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        
        # Templates
        templates = manifest.get("templates") or {}
        # Back-compat: if templates missing, try namingConvention
        if not templates:
            nc = manifest.get("namingConvention") or {}
            templates = {
                "dark": nc.get("inactive", "Dark {member}.png"),
                "light": nc.get("active", "{member}.png"),
                "circle": nc.get("circle", "{member} Circle.png"),
            }
            
        darkTplVar = tk.StringVar(value=templates.get("dark", "Dark {member}.png"))
        lightTplVar = tk.StringVar(value=templates.get("light", "{member}.png"))
        circleTplVar = tk.StringVar(value=templates.get("circle", "{member} Circle.png"))

        ttk.Label(frm, text="Dark template:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=darkTplVar).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))

        ttk.Label(frm, text="Light template:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(frm, textvariable=lightTplVar).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        ttk.Label(frm, text="Circle template:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(frm, textvariable=circleTplVar).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        
        # Age/display order CSV
        currentAgeOrder = manifest.get("ageOrder") or []
        ageCsvDefault = ",".join([str(x).strip() for x in currentAgeOrder if str(x).strip()])
        ageCsvVar = tk.StringVar(value=ageCsvDefault)
        
        ttk.Label(frm, text="Display/Age order CSV (optional):").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=ageCsvVar).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))

        statusVar = tk.StringVar(value="Edit fields, then click Test.")
        ttk.Label(frm, textvariable=statusVar).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

        btnRow = ttk.Frame(frm)
        btnRow.grid(row=6, column=0, columnspan=2, sticky="e", pady=(12, 0))
        
        def onTest():
            try:
                report = self.groupRegistry.testEditGroupOrThrow(
                    groupName=groupName,
                    activeTheme=themeVar.get(),
                    templates={
                        "dark": darkTplVar.get(),
                        "light": lightTplVar.get(),
                        "circle": circleTplVar.get()
                    },
                    ageCsv=ageCsvVar.get()
                )
                statusVar.set(f"OK — {len(report['members'])} members detected.")
                messagebox.showinfo(
                    "Test OK",
                    "Templates and order are valid.\n\n"
                    f"Detected members: {', '.join([m['name'] for m in report['members']])}",
                    parent=win
                )
            except Exception as e:
                statusVar.set("Test failed — see error.")
                messagebox.showerror("Test failed", f"{type(e).__name__}: {e}", parent=win)
                
        def onSave():
            try:
                self.groupRegistry.saveEditGroup(
                    groupName=groupName,
                    activeTheme=themeVar.get(),
                    templates={
                        "dark": darkTplVar.get(),
                        "light": lightTplVar.get(),
                        "circle": circleTplVar.get()
                    },
                    ageCsv=ageCsvVar.get()
                )

                # refresh GUI state
                self.groups = self.groupRegistry.groups
                self.rescanGroups()  # keeps dropdown + members list consistent
                self.currentGroup.set(groupName)
                self.displayMembers(groupName)

                messagebox.showinfo("Saved", f"Updated group '{groupName}'.", parent=win)
                win.destroy()
            except Exception as e:
                messagebox.showerror("Save failed", f"{type(e).__name__}: {e}", parent=win)
                
        ttk.Button(btnRow, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(btnRow, text="Save", command=onSave).pack(side="right", padx=(8, 0))
        ttk.Button(btnRow, text="Test", command=onTest).pack(side="right", padx=(8, 0))
    
    def setStatus(self, message, isError=False):
        if not hasattr(self, "statusVar"):
            return
        self.statusVar.set(message)
        if hasattr(self, "statusLabel") and self.statusLabel:
            self.statusLabel.configure(fg=("red" if isError else "gray"))
            
    def _isAllowedImagePathOrUrl(self, s: str) -> bool:
        # accepts .png/.jpg/.jpeg (either local path or URL path)
        try:
            path = urlparse(s).path.lower()
        except Exception:
            path = (s or "").lower()
        return any(ext in path for ext in (".png", ".jpg", ".jpeg"))
    
    def _normalizeUnicodeUrl(self, url: str) -> str:
        """
        Percent-encode non-ASCII characters in the URL path safely.
        Keeps scheme, domain, query, fragments intact.
        """
        parsed = urlparse(url)

        encodedPath = quote(parsed.path, safe="/%")
        encodedQuery = quote(parsed.query, safe="=&%")

        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            encodedPath,
            parsed.params,
            encodedQuery,
            parsed.fragment
        ))
    
    def _isUrl(self, s: str) -> bool:
        u = urlparse(s)
        return u.scheme in ("http", "https")
    
    def _downloadImageFromUrl(self, url: str) -> Image.Image:
        # Small, safe download to PIL image
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data))
        img.load()  # force decode now so we can catch format errors here
        return img
    
    def _getMemberCustomCachePath(self, groupName: str, memberName: str, ext: str) -> str:
        safe = "".join(c for c in memberName if c.isalnum() or c in (" ", "_", "-")).strip()
        safe = safe.replace(" ", "_") or "member"
        cacheDir = os.path.join("member_images", groupName, "_custom")
        os.makedirs(cacheDir, exist_ok=True)
        return os.path.join(cacheDir, f"{safe}{ext}")
    
    def _saveMemberImageChoice(self, groupName, member, cachePath, imageUrl=None):
        # Persist selection to member dict + JSON
        member["imageCachePath"] = cachePath
        if imageUrl:
            member["imageUrl"] = imageUrl
        else:
            member.pop("imageUrl", None)

        try:
            self.groupRegistry.saveGroupsToJson()
        except Exception:
            # If your registry uses a different save method name, swap it here.
            pass
         
    def loadMemberImage(self, groupName, member):
        cachePath = member.get("imageCachePath")
        
        if cachePath and os.path.exists(cachePath):
            try:
                image = Image.open(cachePath).resize(self.imageSize)
                return ImageTk.PhotoImage(image)
            except Exception:
                pass
        
        imagePath = os.path.join("member_images", groupName, f"{member['name']}.png")
        try:
            image = Image.open(imagePath).resize(self.imageSize)
        except Exception:
            image = Image.new("RGB", self.imageSize, color=member['color'])
        
        return ImageTk.PhotoImage(image)
     
    def openMemberImageMenu(self, event, groupName, member):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="Set image from URL...",
            command=lambda: self.onSetMemberImageFromUrl(groupName, member)
        )
        menu.add_command(
            label="Choose local image...",
            command=lambda: self.onSetMemberImageFromFile(groupName, member)
        )
        menu.add_separator()
        menu.add_command(
            label="Reset to default",
            command=lambda: self.onResetMemberImage(groupName, member)
        )
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def onSetMemberImageFromUrl(self, groupName, member):
        memberName = member.get("name", "member")

        url = simpledialog.askstring(
            "Set member image (URL)",
            f"Paste a direct image URL for {memberName} (.png/.jpg/.jpeg):",
            parent=self.root
        )
        if url is None:
            self.setStatus("Image update cancelled.")
            return

        url = self._normalizeUnicodeUrl(url.strip())
        if not url or not self._isUrl(url) or not self._isAllowedImagePathOrUrl(url):
            msg = "Please paste a direct http(s) URL ending in .png/.jpg/.jpeg."
            self.setStatus(msg, isError=True)
            messagebox.showerror("Invalid image URL", msg, parent=self.root)
            return

        try:
            img = self._downloadImageFromUrl(url)

            # decide extension based on URL path
            urlPath = urlparse(url).path.lower()
            ext = ".png" if urlPath.endswith(".png") else ".jpg"

            cachePath = self._getMemberCustomCachePath(groupName, memberName, ext)
            img.save(cachePath)

            self._saveMemberImageChoice(groupName, member, cachePath, imageUrl=url)
            self.setStatus(f"Updated image for {memberName}.")
            self.displayMembers(groupName)

        except Exception as e:
            msg = f"Failed to load image from URL: {type(e).__name__}: {e}"
            self.setStatus(msg, isError=True)
            messagebox.showerror("Image download failed", msg, parent=self.root)

    def onSetMemberImageFromFile(self, groupName, member):
        memberName = member.get("name", "member")

        filePath = filedialog.askopenfilename(
            parent=self.root,
            title=f"Choose image for {memberName}",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not filePath:
            self.setStatus("Image update cancelled.")
            return

        if not self._isAllowedImagePathOrUrl(filePath):
            msg = "Invalid file format. Please choose a .png/.jpg/.jpeg image."
            self.setStatus(msg, isError=True)
            messagebox.showerror("Invalid image file", msg, parent=self.root)
            return

        try:
            # Validate it’s actually an image
            img = Image.open(filePath)
            img.load()

            ext = os.path.splitext(filePath)[1].lower()
            if ext not in (".png", ".jpg", ".jpeg"):
                msg = "Invalid file extension. Please use .png/.jpg/.jpeg."
                self.setStatus(msg, isError=True)
                messagebox.showerror("Invalid image file", msg, parent=self.root)
                return

            # Cache it inside your project so it won’t break if user moves the original file
            cachePath = self._getMemberCustomCachePath(groupName, memberName, ext if ext != ".jpeg" else ".jpg")
            shutil.copyfile(filePath, cachePath)

            self._saveMemberImageChoice(groupName, member, cachePath, imageUrl=None)
            self.setStatus(f"Updated image for {memberName}.")
            self.displayMembers(groupName)

        except Exception as e:
            msg = f"Failed to load that image: {type(e).__name__}: {e}"
            self.setStatus(msg, isError=True)
            messagebox.showerror("Image load failed", msg, parent=self.root)

    def onResetMemberImage(self, groupName, member):
        memberName = member.get("name", "member")
        member.pop("imageUrl", None)
        member.pop("imageCachePath", None)

        try:
            self.groupRegistry.saveGroupsToJson()
        except Exception:
            pass

        self.setStatus(f"Reset image for {memberName} to default.")
        self.displayMembers(groupName)
  
    def openSongAlbumMenu(self, event, groupName, songName):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="Assign to existing album...",
            command=lambda: self.promptAssignSongToAlbum(groupName, songName)
        )
        menu.add_command(
            label="Create new album and assign...",
            command=lambda: self.promptCreateAlbumAndAssign(groupName, songName)
        )
        menu.add_separator()
        menu.add_command(
            label="Set album art for this song’s album...",
            command=lambda: self.promptSetAlbumArtForSongsAlbum(groupName, songName)
        )
        menu.add_command(
            label="Remove album assignment for this song",
            command=lambda: self.clearSongAlbumAssignment(groupName, songName)
        )

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
      
    def promptAssignSongToAlbum(self, groupName, songName):
        parent = self.getActiveDialogParent()
        albums = self.groupRegistry.getAlbums(groupName)  # dict: albumId -> meta
        if not albums:
            messagebox.showinfo("No albums", "No albums exist yet. Create one first.", parent=parent)
            return

        # Stable ordering for indices
        albumIds = sorted(albums.keys())

        # Show: "0: ive_switch"
        albumList = "\n".join([f"{i}: {albumId}" for i, albumId in enumerate(albumIds)])
        userInput = simpledialog.askstring(
            "Assign album",
            f"Enter an album index or id for:\n{songName}\n\nExisting albums:\n{albumList}",
            parent=parent
        )
        if not userInput:
            return

        userInput = userInput.strip()

        # Allow numeric index
        albumId = None
        if userInput.isdigit():
            idx = int(userInput)
            if 0 <= idx < len(albumIds):
                albumId = albumIds[idx]
            else:
                messagebox.showerror(
                    "Invalid album",
                    f"Index out of range: {idx}\nValid range: 0 to {len(albumIds) - 1}",
                    parent=parent
                )
                return
        else:
            # Allow direct album id
            if userInput in albums:
                albumId = userInput
            else:
                messagebox.showerror("Invalid album", f"Album id not found: {userInput}", parent=parent)
                return

        self.groupRegistry.setSongAlbum(groupName, songName, albumId)
        self.setStatus(f"Assigned '{songName}' to album '{albumId}'.")
        self.refreshSongPickerUI()
    
    def promptAssignSongToAlbum(self, groupName, songName):
        parent = self.getActiveDialogParent()
        albums = self.groupRegistry.getAlbums(groupName)  # dict: albumId -> meta
        if not albums:
            messagebox.showinfo(
                "No albums",
                "No albums exist yet. Create one first.",
                parent=parent
            )
            return

        # Stable, readable ordering
        albumIds = sorted(albums.keys())

        # Build numbered display
        lines = []
        for idx, albumId in enumerate(albumIds, start=1):
            displayName = albums[albumId].get("displayName", albumId)
            if displayName != albumId:
                lines.append(f"{idx}) {albumId}  ({displayName})")
            else:
                lines.append(f"{idx}) {albumId}")

        albumListText = "\n".join(lines)

        userInput = simpledialog.askstring(
            "Assign album",
            (
                f"Assign '{songName}' to an album.\n\n"
                f"You may enter either:\n"
                f"• the album number (e.g. 1)\n"
                f"• the album id (e.g. love_tune)\n\n"
                f"Available albums:\n{albumListText}"
            ),
            parent=parent
        )

        if not userInput:
            return

        userInput = userInput.strip()

        resolvedAlbumId = None

        # Case 1: numeric index
        if userInput.isdigit():
            idx = int(userInput)
            if 1 <= idx <= len(albumIds):
                resolvedAlbumId = albumIds[idx - 1]
            else:
                messagebox.showerror(
                    "Invalid album number",
                    f"Album number must be between 1 and {len(albumIds)}.",
                    parent=parent
                )
                return

        # Case 2: album id string
        else:
            if userInput in albums:
                resolvedAlbumId = userInput
            else:
                messagebox.showerror(
                    "Invalid album",
                    f"Album id not found: {userInput}",
                    parent=parent
                )
                return

        # Assign + persist
        self.groupRegistry.setSongAlbum(groupName, songName, resolvedAlbumId)
        self.setStatus(f"Assigned '{songName}' to album '{resolvedAlbumId}'.")
        self.refreshSongPickerUI()
        
    def promptCreateAlbumAndAssign(self, groupName, songName):
        parent = self.getActiveDialogParent()
        displayName = simpledialog.askstring(
            "Create album",
            "Enter album name (display name):",
            parent=parent
        )
        if not displayName:
            return

        # create a stable id (lower + underscores); you can tweak this
        albumId = "".join(c.lower() if c.isalnum() else "_" for c in displayName).strip("_")
        if not albumId:
            albumId = "album"

        albums = self.groupRegistry.getAlbums(groupName)
        if albumId in albums:
            # simple collision resolver
            i = 2
            base = albumId
            while f"{base}_{i}" in albums:
                i += 1
            albumId = f"{base}_{i}"

        self.groupRegistry.createAlbum(groupName, albumId, displayName=displayName)
        self.groupRegistry.setSongAlbum(groupName, songName, albumId)
        self.setStatus(f"Created album '{displayName}' and assigned '{songName}'.")
        self.refreshSongPickerUI()
        
    def clearSongAlbumAssignment(self, groupName, songName):
        # simplest: remove mapping, and remove from any album songs list
        g = self.groupRegistry.groups.get(groupName, {})
        if not isinstance(g, dict):
            return

        prev = g.get("songToAlbum", {}).pop(songName, None)
        if prev:
            albums = g.get("albums", {})
            if prev in albums and songName in albums[prev].get("songs", []):
                albums[prev]["songs"].remove(songName)

        self.groupRegistry.saveGroupsToJson()
        self.setStatus(f"Cleared album assignment for '{songName}'.")
        self.refreshSongPickerUI()
        
    def promptSetAlbumArtForSongsAlbum(self, groupName, songName):
        parent = self.getActiveDialogParent()
        albumId = self.groupRegistry.getSongAlbumId(groupName, songName)
        if not albumId:
            messagebox.showinfo("No album", "This song has no album assigned yet.", parent=parent)
            return
        self.openAlbumArtMenu(groupName, albumId)

    def openAlbumArtMenu(self, groupName, albumId):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="Set album art from URL...",
            command=lambda: self.onSetAlbumArtFromUrl(groupName, albumId)
        )
        menu.add_command(
            label="Choose local album art...",
            command=lambda: self.onSetAlbumArtFromFile(groupName, albumId)
        )
        menu.add_separator()
        menu.add_command(
            label="Reset album art",
            command=lambda: self.onResetAlbumArt(groupName, albumId)
        )

        # show menu centered-ish (no event here), so pop it near mouse:
        x = self.root.winfo_pointerx()
        y = self.root.winfo_pointery()
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()
            
    def _getAlbumArtCachePath(self, groupName, albumId, ext):
        cacheDir = os.path.join("member_images", groupName, "_album_art")
        os.makedirs(cacheDir, exist_ok=True)
        return os.path.join(cacheDir, f"{albumId}{ext}")

    def onSetAlbumArtFromUrl(self, groupName, albumId):
        parent = self.getActiveDialogParent()
        url = simpledialog.askstring(
            "Set album art (URL)",
            "Paste a direct image URL (.png/.jpg/.jpeg):",
            parent=parent
        )
        if not url:
            return
        url = self._normalizeUnicodeUrl(url.strip())

        if not self._isUrl(url) or not self._isAllowedImagePathOrUrl(url):
            messagebox.showerror("Invalid URL", "Must be a direct http(s) URL ending in .png/.jpg/.jpeg.", parent=parent)
            return

        try:
            img = self._downloadImageFromUrl(url)
            ext = ".png" if urlparse(url).path.lower().endswith(".png") else ".jpg"
            cachePath = self._getAlbumArtCachePath(groupName, albumId, ext)
            img.save(cachePath)

            self.groupRegistry.setAlbumArt(groupName, albumId, cachePath=cachePath, url=url)
            self.setStatus(f"Updated album art for '{albumId}'.")
            self.refreshSongPickerUI()
        except Exception as e:
            messagebox.showerror("Album art failed", f"{type(e).__name__}: {e}", parent=parent)

    def onSetAlbumArtFromFile(self, groupName, albumId):
        parent = self.getActiveDialogParent()
        filePath = filedialog.askopenfilename(
            parent=parent,
            title="Choose album art image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not filePath:
            return

        if not self._isAllowedImagePathOrUrl(filePath):
            messagebox.showerror("Invalid file", "Must be .png/.jpg/.jpeg.", parent=parent)
            return

        try:
            img = Image.open(filePath)
            img.load()

            ext = os.path.splitext(filePath)[1].lower()
            if ext == ".jpeg":
                ext = ".jpg"
            if ext not in (".png", ".jpg"):
                messagebox.showerror("Invalid file", "Must be .png/.jpg/.jpeg.", parent=parent)
                return

            cachePath = self._getAlbumArtCachePath(groupName, albumId, ext)
            shutil.copyfile(filePath, cachePath)

            self.groupRegistry.setAlbumArt(groupName, albumId, cachePath=cachePath, url="")
            self.setStatus(f"Updated album art for '{albumId}'.")
            self.refreshSongPickerUI()
        except Exception as e:
            messagebox.showerror("Album art failed", f"{type(e).__name__}: {e}", parent=parent)

    def onResetAlbumArt(self, groupName, albumId):
        self.groupRegistry.setAlbumArt(groupName, albumId, cachePath="", url="")
        self.setStatus(f"Reset album art for '{albumId}'.")
        self.refreshSongPickerUI()
      
    def trainModel(self):
        selectedGroup = self.currentGroup.get()
        memberList = [member['name'] for member in self.groups[selectedGroup]]
        print(f"Starting to train on vocals forr {selectedGroup}")
        # train(selectedGroup, memberList)
    
    def enableGlobalScroll(self, canvas, container):
        def _onMouseWheel(event):
            # For Windows and MacOS
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        def _onMouseWheelLinux(event):
            # For Linux systems (Button-4 = up, Button-5 = down)
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        # Bind when mouse enters
        container.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _onMouseWheel))
        container.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.bind("<Button-4>", _onMouseWheelLinux)  # Linux scroll up
        canvas.bind("<Button-5>", _onMouseWheelLinux)  # Linux scroll down
        
    def getAlbumForSong(self, groupName, songName):
        # returns albumId (or None) using registry data
        albumId = self.groupRegistry.getSongAlbumId(groupName, songName)
        return albumId if albumId else None
    
    def initPlaceholders(self):
        try:
            img = Image.open("placeholder.png").resize((100, 100))
        except Exception as e:
            print(f"[⚠️] Failed to load placeholder.png: {e}")
            img = Image.new("RGB", (100, 100), "gray")

        self.placeholderIcon = ImageTk.PhotoImage(img)
    
    def chooseSongWindow(self, selectedGroup, title, callback):
        """
        Opens (or reuses) a scrollable window with all available songs in the group.
        When a song is clicked, callback(songName) is triggered.
        """

        # Store context so refreshSongPickerUI can rebuild consistently
        self.songPickerSelectedGroup = selectedGroup
        self.songPickerTitle = title
        self.songPickerCallback = callback

        # If already open, just refresh + raise it
        if hasattr(self, "songPickerWindow") and self.songPickerWindow is not None:
            try:
                if self.songPickerWindow.winfo_exists():
                    self.songPickerWindow.title(title)
                    self.refreshSongPickerUI()
                    self.songPickerWindow.lift()
                    return
            except Exception:
                pass

        self.albumImageRefs = []

        songWindow = tk.Toplevel(self.root)
        self.songPickerWindow = songWindow
        songWindow.title(title)
        songWindow.geometry(self.root.winfo_geometry())

        # ---- Layout: top bar + scrollable list below ----
        top = tk.Frame(songWindow)
        top.pack(fill="x", padx=8, pady=6)

        listContainer = tk.Frame(songWindow)
        listContainer.pack(fill="both", expand=True)

        canvas = tk.Canvas(listContainer, highlightthickness=0, bd=0, relief="flat")
        scrollbar = tk.Scrollbar(listContainer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = tk.Frame(canvas)
        windowId = canvas.create_window((0, 0), window=frame, anchor="nw")
        self.songPickerFrameWindowId = windowId

        # Make the embedded frame always match canvas width -> no dead unscrollable margins
        def _onCanvasConfigure(e):
            try:
                canvas.itemconfig(windowId, width=e.width)
            except Exception:
                pass

        canvas.bind("<Configure>", _onCanvasConfigure)

        # Update scrollregion whenever content changes size
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Store references for refresh
        self.songPickerCanvas = canvas
        self.songPickerFrame = frame

        # Ensure scroll works no matter what widget your cursor is over
        self._bindScrollEverywhere(songWindow, canvas)

        # If the user closes it, clear references so the next open recreates cleanly
        def _onClose():
            try:
                songWindow.destroy()
            finally:
                self.songPickerWindow = None
                self.songPickerCanvas = None
                self.songPickerFrame = None
                self.songPickerFrameWindowId = None

        songWindow.protocol("WM_DELETE_WINDOW", _onClose)

        # --- song picker: selected background / MV video ---
        self.songPickerVideoPathVar = tk.StringVar(value="")
        self.songPickerIsMusicVideoVar = tk.BooleanVar(value=False)

        videoLabel = tk.Label(top, text="Video: (none)", anchor="w")
        videoLabel.pack(side="left", fill="x", expand=True)

        def _setVideoLabel():
            vp = self.songPickerVideoPathVar.get().strip()
            if not vp:
                videoLabel.config(text="Video: (none)")
            else:
                videoLabel.config(text=f"Video: {os.path.basename(vp)}")

        def onChooseVideo():
            songDir = self.groupRegistry.getGroupMediaDir(self.songPickerSelectedGroup)
            chosen = filedialog.askopenfilename(
                parent=songWindow,
                title="Choose a video file (.mp4/.mov/.mkv)",
                initialdir=songDir if os.path.isdir(songDir) else ".",
                filetypes=[
                    ("Video files", "*.mp4 *.mov *.mkv *.webm"),
                    ("MP4", "*.mp4"),
                    ("All files", "*.*"),
                ],
            )
            if not chosen:
                return
            self.songPickerVideoPathVar.set(chosen)
            _setVideoLabel()

        tk.Button(top, text="Choose Video…", command=onChooseVideo).pack(side="right", padx=(8, 0))

        # First render
        self.refreshSongPickerUI()
        _setVideoLabel()
    
    
    def refreshSongPickerUI(self):
        """
        Rebuilds the song list UI inside the existing song picker window.
        Safe to call after changing albums / album art.
        """
        if not hasattr(self, "songPickerWindow") or self.songPickerWindow is None:
            return
        if not self.songPickerWindow.winfo_exists():
            return

        selectedGroup = getattr(self, "songPickerSelectedGroup", None)
        title = getattr(self, "songPickerTitle", "Choose Song")
        callback = getattr(self, "songPickerCallback", None)
        if not selectedGroup or callback is None:
            return

        self.songPickerWindow.title(title)

        frame = self.songPickerFrame
        canvas = self.songPickerCanvas

        # Preserve scroll position (roughly)
        try:
            yview = canvas.yview()
        except Exception:
            yview = (0.0, 1.0)

        # Clear existing rows
        for w in frame.winfo_children():
            w.destroy()

        # Recompute song list
        songDir = self.groupRegistry.getGroupMediaDir(selectedGroup)
        try:
            songList, bestBySong = findBestAudioFiles(songDir)
        except Exception as e:
            print(f"❌ Could not list songs in {songDir}: {e}")
            songList = []

        if not songList:
            msg = tk.Label(frame, text="❌ No songs available.", font=("Helvetica", 14), fg="red")
            msg.pack(anchor="w", padx=10, pady=10)
            canvas.configure(scrollregion=canvas.bbox("all"))
            return

        # Keep image refs alive
        self.albumImageRefs = []

        albums = self.groupRegistry.getAlbums(selectedGroup) or {}

        # ---- Group songs by album, and collect truly-unclaimed ones ----
        defaultAlbumId = self._getDefaultAlbumId(albums)

        albumToSongs = {}   # albumId -> [songName...]
        unclaimed = []

        for songName in songList:
            albumId = self.getAlbumForSong(selectedGroup, songName)

            # If missing / invalid, treat as unclaimed (failsafe section)
            if (not albumId) or (albumId not in albums):
                # If you *want* missing mappings to still fall into the default album, flip this:
                albumId = defaultAlbumId if defaultAlbumId in albums else None
                if albumId is None:
                    unclaimed.append(songName)
                continue

            albumToSongs.setdefault(albumId, []).append(songName)

        # Sort albums by displayName then id (stable, human-friendly)
        def _albumSortKey(aid: str):
            meta = albums.get(aid, {})
            display = (meta.get("displayName") or aid).strip()
            return (display.lower(), aid.lower())

        orderedAlbumIds = sorted(albumToSongs.keys(), key=_albumSortKey)

        # ---- Render UI sections ----
        headerFont = ("Helvetica", 16, "bold")
        sectionPadY = 10

        for albumId in orderedAlbumIds:
            meta = albums.get(albumId, {})
            displayName = (meta.get("displayName") or albumId).strip()

            hdr = tk.Label(frame, text=displayName, font=headerFont, anchor="w")
            hdr.pack(fill="x", padx=10, pady=(sectionPadY, 6))

            songs = sorted(albumToSongs.get(albumId, []), key=str.lower)
            for songName in songs:
                self._addSongRow(frame, selectedGroup, songName, albums, callback)

        if unclaimed:
            hdr = tk.Label(frame, text="Unclaimed", font=headerFont, anchor="w", fg="#aa2222")
            hdr.pack(fill="x", padx=10, pady=(sectionPadY, 6))

            for songName in sorted(unclaimed, key=str.lower):
                self._addSongRow(frame, selectedGroup, songName, albums, callback)

        # Update scroll region and restore scroll
        canvas.configure(scrollregion=canvas.bbox("all"))
        try:
            canvas.yview_moveto(yview[0])
        except Exception:
            pass

        # Rebind scroll to everything we just created (so buttons/labels don't “eat” scrolling)
        self._bindScrollEverywhere(self.songPickerWindow, canvas)
    
    def _getDefaultAlbumId(self, albumsDict):
        """
        Failsafe default album. If you store a default album id in your registry later,
        read it here. For now we try common keys, then fall back to "defaultTheme".
        """
        # If you later add group-level config like groups[group]["defaultAlbumId"],
        # plug it in here.
        for candidate in ("defaultTheme", "default", "main"):
            if candidate in (albumsDict or {}):
                return candidate
        # If there is at least one album, pick the first alphabetically as fallback
        if albumsDict:
            return sorted(albumsDict.keys(), key=str.lower)[0]
        return "defaultTheme"


    def _addSongRow(self, parentFrame, selectedGroup, songName, albums, callback):
        def _onPickSong(name):
            videoPath = self.songPickerVideoPathVar.get().strip() or None
            try:
                self.songPickerWindow.destroy()
            except Exception:
                pass
            callback(name, videoPath)

        songFrame = tk.Frame(parentFrame, pady=0)
        songFrame.pack(fill="x", padx=5)

        songIcon = self.placeholderIcon  # default

        albumId = self.getAlbumForSong(selectedGroup, songName)
        if albumId and albumId in (albums or {}):
            album = albums.get(albumId, {})
            albumArtPath = album.get("albumArtCachePath", "")
            if albumArtPath and os.path.exists(albumArtPath):
                try:
                    image = Image.open(albumArtPath).resize((100, 100))
                    songIcon = ImageTk.PhotoImage(image)
                except Exception as e:
                    print(f"[⚠️] Couldn't load album art {albumArtPath}: {e}")
                    songIcon = self.placeholderIcon

        self.albumImageRefs.append(songIcon)

        imgLabel = tk.Label(songFrame, image=songIcon)
        imgLabel.pack(side="left", padx=5)

        imgLabel.bind(
            "<Button-3>",
            lambda e, g=selectedGroup, s=songName: self.openSongAlbumMenu(e, g, s)
        )

        button = tk.Button(
            songFrame,
            text=songName,
            font=("Helvetica", 14),
            anchor="w",
            justify="left",
            command=lambda name=songName: _onPickSong(name),
        )
        button.pack(side="left", fill="x", expand=True)


    def _bindScrollEverywhere(self, rootWidget, canvas):
        """
        Make mouse-wheel scrolling work no matter what child widget the mouse is over.
        This fixes the “only some areas scroll” issue (buttons/labels often swallow it).
        """
        def _onMouseWheel(event):
            # Windows / macOS
            try:
                canvas.yview_scroll(-1 * int(event.delta / 120), "units")
            except Exception:
                pass
            return "break"

        def _onMouseWheelLinux(event):
            # Linux: Button-4 up, Button-5 down
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            return "break"

        # Bind to the toplevel AND also bind_all while this window has focus/hover.
        # Using bind_all is the “I don’t care what widget you’re on” solution.
        try:
            rootWidget.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _onMouseWheel))
            rootWidget.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
            rootWidget.bind("<Button-4>", _onMouseWheelLinux)
            rootWidget.bind("<Button-5>", _onMouseWheelLinux)
        except Exception:
            pass
    
    # -------- Cache helpers --------   
    def getCacheDir(self) -> Path:
        # Always use ./cache_audio
        return Path("./cache_audio")
    
    def getCacheSizeBytes(self) -> int:
        cacheDir = self.getCacheDir()
        if not cacheDir.exists():
            return 0

        total = 0
        try:
            for root, _, files in os.walk(cacheDir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    try:
                        total += os.path.getsize(fp)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Error getting size: {e}")
            return 0
        
        return total
    
    def formatCacheSize(self, numBytes: int) -> tuple[str, str]:
        # Returns (valueStr, unitStr) where unit is MB or GB.
        if numBytes <= 0:
            return ("0.00", "MB")

        mb = numBytes / (1024 * 1024)
        if mb < 1024:
            return (f"{mb:.2f}", "MB")

        gb = mb / 1024
        return (f"{gb:.2f}", "GB")
    
    def refreshCacheMenuLabel(self):
        if not self.cacheMenu:
            return

        sizeBytes = self.getCacheSizeBytes()
        valueStr, unitStr = self.formatCacheSize(sizeBytes)

        # label format: "Clear cache: {size in mb/gb} {mb/gb}"
        self.cacheMenu.entryconfig(0, label=f"Clear cache: {valueStr} {unitStr}")
        
    def onClearCache(self):
        parent = self.getActiveDialogParent()

        cacheDir = self.getCacheDir()
        sizeBytes = self.getCacheSizeBytes()
        valueStr, unitStr = self.formatCacheSize(sizeBytes)

        if sizeBytes == 0:
            messagebox.showinfo(
                "Cache",
                f"Cache is already empty.\n\nFolder: {cacheDir.resolve()}",
                parent=parent
            )
            self.refreshCacheMenuLabel()
            return

        ok = messagebox.askokcancel(
            "Clear cache",
            (
                f"You are about to delete all cached audio in:\n{cacheDir.resolve()}\n\n"
                f"Size: {valueStr} {unitStr}\n\n"
                "This cannot be undone. Continue?"
            ),
            icon="warning",
            parent=parent
        )
        if not ok:
            return

        try:
            if cacheDir.exists():
                shutil.rmtree(cacheDir)
            cacheDir.mkdir(parents=True, exist_ok=True)

            self.setStatus(f"Cleared cache: {valueStr} {unitStr}.")
            messagebox.showinfo(
                "Cache cleared",
                f"Deleted {valueStr} {unitStr} from cache.\n\nFolder: {cacheDir.resolve()}",
                parent=parent
            )
        except Exception as e:
            msg = f"Failed to clear cache: {type(e).__name__}: {e}"
            self.setStatus(msg, isError=True)
            messagebox.showerror("Clear cache failed", msg, parent=parent)
        finally:
            self.refreshCacheMenuLabel()
        
    def selectSong(self):
        selectedGroup = self.currentGroup.get()
        songDir = self.groupRegistry.getGroupMediaDir(selectedGroup)
        #print(f"Current songDir: {songDir}")
        modelPath = f"./models/{selectedGroup}_muq_head.pt"
        
        def onSongPicked(songName, videoPath):
            groupDir = self.groupRegistry.getGroupDir(selectedGroup)
            groupManifest = self.groupRegistry._loadGroupManifest(groupDir)
            memberImages = self.groupRegistry.loadMemberImages(selectedGroup, groupManifest, songName)
            video = videoPath if videoPath and os.path.exists(videoPath) else None
            launchVoiceApp(songName, memberImages, video)
            
        def launchVoiceApp(songName, memberImages, videoPath):
            if not ModalGuard.try_open("voice_app"):
                return  # another modal is open
            testSongPath = pickBestAudioForStem(songDir, songName)
            vocalsOnlyPath = os.path.join(songDir, f"{songName}_vocals.wav")
            vocalsLeadPath = os.path.join(songDir, f"{songName}_leading_vocals.wav")
            vocalsBackingPath =  os.path.join(songDir, f"{songName}_backing_vocals.wav")
            
            if not os.path.exists(vocalsOnlyPath):
                print(f"⚠️ Vocals-only file not found for {songName}")
                vocalsOnlyPath = testSongPath  # fallback

            appWindow = tk.Toplevel(self.root)
            appWindow.title("Line Distribution Labeler")
            appWindow.geometry("960x540")
            
            # Please stop having a stroke app
            appWindow.update_idletasks()
            appWindow.minsize(960, 540)
            continueApp = [True]
            firstMember = self.groups[selectedGroup]['members'][0]['name']
            app = None

            def onClose():
                if tk.messagebox.askyesno("Exit", "Do you want to stop the application?"):
                    if hasattr(app, "videoTrackItem") and app.videoTrackItem:
                        app.videoTrackItem.pause()
                        app.videoTrackItem.stop()

                    continueApp[0] = False
                    appWindow.destroy()
                    
                    sys.exit()

            appWindow.protocol("WM_DELETE_WINDOW", onClose)

            memberList = self.groups[selectedGroup]["members"]
            app = VoiceDetectionApp(
                root=appWindow,
                members=memberList,
                modelPath=modelPath,
                images=memberImages,
                testSongPath=testSongPath,
                vocalsOnlyPath=vocalsOnlyPath,
                vocalsLeadPath=vocalsLeadPath,
                vocalsBackingPath=vocalsBackingPath,
                videoPath=videoPath,
                selectedGroup=selectedGroup,
                songDir=songDir
            )
            if hasattr(app, "videoTrackItem") and app.videoTrackItem and app.videoTrackItem.thread:
                app.videoTrackItem.thread.daemon = True

            root.mainloop()

        # ✅ Launch song picker
        self.chooseSongWindow(
            selectedGroup=selectedGroup,
            title=f"Choose a song for {selectedGroup}",
            callback=onSongPicked
        )    
    
    def openGenerateMemberImagesDialog(self):
        groupName = self.currentGroup.get().strip()
        if not groupName:
            messagebox.showerror("Generate Images", "No group selected.", parent=self.root)
            return

        # Load group manifest (group.json) from ./group_icons/<group>/group.json
        groupDir = self.groupRegistry.getGroupDir(groupName)
        manifest = self.groupRegistry._loadGroupManifest(groupDir)
        if not manifest:
            messagebox.showerror(
                "Generate Images",
                f"No group.json found for '{groupName}'.\n\n"
                "Create or rescan the group first.",
                parent=self.root
            )
            return

        win = tk.Toplevel(self.root)
        win.title(f"Generate Member Images — {groupName}")
        win.transient(self.root)
        win.grab_set()

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        # Directory that contains "{member} Circle.png"
        dirVar = tk.StringVar(value="")
        # Square color (background block behind circle)
        squareVar = tk.StringVar(value="#1A1A1A")   # Bright square color (user)
        ringVar = tk.StringVar(value="#ffffff")     # Bright ring override (user)
        # Font path (you can default to your Hiragino path if you want)
        fontVar = tk.StringVar(
            value=r"C:\Users\elvin\AppData\Local\Microsoft\Windows\Fonts\Hiragino Sans GB W3.ttf"
        )

        ttk.Label(frm, text="Circle images folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=dirVar).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        def chooseDir():
            selected = filedialog.askdirectory(
                parent=win,
                title="Select folder containing '{member} Circle.png' files"
            )
            if selected:
                dirVar.set(selected)

        ttk.Button(frm, text="Browse...", command=chooseDir).grid(row=0, column=2, padx=(8, 0))

        def pickHexColor(targetVar: tk.StringVar, title: str):
            initial = targetVar.get().strip() or "#ffffff"
            rgb, hexStr = colorchooser.askcolor(color=initial, title=title, parent=win)
            if hexStr:
                targetVar.set(hexStr.lower())
        
        # Bright square color
        ttk.Label(frm, text="Bright square color (hex):").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=squareVar).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))
        ttk.Button(frm, text="Pick…", command=lambda: pickHexColor(squareVar, "Pick bright square color")).grid(
            row=1, column=2, padx=(8, 0), pady=(10, 0)
        )

        # Bright ring color override
        ttk.Label(frm, text="Bright ring color (hex):").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=ringVar).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))
        ttk.Button(frm, text="Pick…", command=lambda: pickHexColor(ringVar, "Pick bright ring color")).grid(
            row=2, column=2, padx=(8, 0), pady=(10, 0)
        )

        ttk.Label(frm, text="Font file (.ttf/.otf):").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=fontVar).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))

        def chooseFont():
            selected = filedialog.askopenfilename(
                parent=win,
                title="Select font file",
                filetypes=[("Font files", "*.ttf *.otf"), ("All files", "*.*")]
            )
            if selected:
                fontVar.set(selected)

        ttk.Button(frm, text="Browse...", command=chooseFont).grid(row=3, column=2, padx=(8, 0), pady=(10, 0))

        statusVar = tk.StringVar(value="Pick a folder containing circle PNGs, then click Create.")
        ttk.Label(frm, textvariable=statusVar).grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 0))

        btnRow = ttk.Frame(frm)
        btnRow.grid(row=5, column=0, columnspan=3, sticky="e", pady=(12, 0))

        def onCreate():
            outDir = dirVar.get().strip()
            squareColor = squareVar.get().strip()
            ringColor = ringVar.get().strip()
            fontPath = fontVar.get().strip()

            if not outDir or not os.path.isdir(outDir):
                messagebox.showerror("Generate Images", "Please choose a valid folder.", parent=win)
                return

            if not fontPath or not os.path.exists(fontPath):
                messagebox.showerror("Generate Images", "Please choose a valid font file path.", parent=win)
                return

            try:
                createdBright, createdDark, missing = self.generateMemberImagesFromManifest(
                    groupName=groupName,
                    manifest=manifest,
                    circlesDir=outDir,
                    outputDir=outDir,
                    squareColorY=squareColor,
                    ringOverride=ringColor,
                    fontPath=fontPath
                )

                statusVar.set(f"Done. Bright: {createdBright}, Dark: {createdDark}, Missing circles: {missing}")

                messagebox.showinfo(
                    "Generate Images",
                    f"Success for {groupName}.\n\n"
                    f"Created bright images: {createdBright}\n"
                    f"Created dark images: {createdDark}\n"
                    f"Missing circle images: {missing}\n\n"
                    f"Output folder:\n{outDir}",
                    parent=win
                )

                # Optional: if your GUI displays from member_images/<group>, you might copy outputs there.
                # For now, we just generate into chosen folder as requested.

            except Exception as e:
                messagebox.showerror("Generate Images Failed", f"{type(e).__name__}: {e}", parent=win)

        ttk.Button(btnRow, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(btnRow, text="Create", command=onCreate).pack(side="right", padx=(8, 0))
    
    def generateMemberImagesFromManifest(
        self,
        groupName: str,
        manifest: dict,
        circlesDir: str,
        outputDir: str,
        squareColorY: str,
        ringOverride: str,
        fontPath: str
    ):
        """
        Uses group manifest format like TWICE example:
        manifest["members"] = [{"name": "...", "color": "#rrggbb"}, ...]
        manifest["templates"] optionally specifies naming templates.
        Searches for "{member} Circle.png" (or template-driven) in circlesDir.
        Writes bright + dark outputs into outputDir.
        """
        templates = manifest.get("templates") or {}
        circleTpl = templates.get("circle", "{member} Circle.png")
        lightTpl = templates.get("light", "{member}.png")
        darkTpl = templates.get("dark", "Dark {member}.png")

        members = manifest.get("members", [])
        if not isinstance(members, list):
            raise ValueError("Manifest 'members' must be a list.")

        createdBright = 0
        createdDark = 0
        missing = 0

        os.makedirs(outputDir, exist_ok=True)

        for m in members:
            if not isinstance(m, dict):
                continue

            memberName = (m.get("name") or "").strip()
            memberColor = (m.get("color") or "").strip()

            if not memberName:
                continue
            if not memberColor:
                memberColor = "#ffffff"  # fallback

            circleFilename = circleTpl.format(member=memberName)
            circlePath = os.path.join(circlesDir, circleFilename)

            if not os.path.exists(circlePath):
                print(f"[WARN] Missing circle image for {groupName}/{memberName}: {circleFilename}")
                missing += 1
                continue

            # Bright
            make_member_card(
                member_name=memberName,
                color_x=ringOverride,
                color_y=squareColorY,
                circles_dir=circlesDir,
                output_dir=outputDir,
                output_path=lightTpl.format(member=memberName),
                font_path=fontPath,
            )
            createdBright += 1

            # Dark
            make_dark_member_card(
                member_name=memberName,
                color_y=memberColor,
                circles_dir=circlesDir,
                output_dir=outputDir,
                output_path=darkTpl.format(member=memberName),
                font_path=fontPath,
            )
            createdDark += 1

        return createdBright, createdDark, missing
    
    def openSetMediaFolderDialog(self):
        groupName = self.currentGroup.get().strip()
        if not groupName:
            messagebox.showerror("Set Media Folder", "No group selected.", parent=self.root)
            return
        
        # Start in current folder if set, else default
        currentDir = self.groupRegistry.getGroupMediaDir(groupName)
        initialDir = currentDir if os.path.isdir(currentDir) else os.path.join(".", "training_data", groupName)

        selected = filedialog.askdirectory(
            parent=self.root,
            title=f"Select media folder for {groupName}",
            initialdir=initialDir
        )
        if not selected:
            return

        self.groupRegistry.setGroupMediaDir(groupName, selected)
        
        messagebox.showinfo(
            "Media Folder Set",
            f"Media folder for {groupName} is now:\n{selected}\n\n"
            "Put all media files for this group in that folder (e.g. .mp3, .wav, .mp4, etc).",
            parent=self.root
        )
    
    def visualizeMemberVocals(self):
        from sklearn.metrics import adjusted_rand_score
        from sklearn.cluster import KMeans
        
        selectedGroup = self.currentGroup.get()
        
        
    def combineAllVocalsFromGroup(self):
        currentGroup = self.currentGroup.get()
        labelDir = f"./saved_labels/{currentGroup}"
        audioDir = f"./training_data/{currentGroup}"
        
        if not os.path.exists(labelDir) or not os.path.exists(audioDir):
            messagebox.showerror("Error", f"Paths not found for group: {currentGroup}")
            return
        
        # Get all JsON label files
        jsonFiles = [
            os.path.join(labelDir, f) for f in os.listdir(labelDir)
            if f.endswith("_labels.json")
        ]
        
        # Get all '_vocals.wav' files
        vocalsOnlySongs = [
            f for f in os.listdir(audioDir)
            if f.endswith("_vocals.wav")
            and not f.endswith("_leading_vocals.wav")
            and not f.endswith("_backing_vocals.wav")
        ]
        
        if not jsonFiles or not vocalsOnlySongs:
            messagebox.showwarning("No Data", "No vocal or label files found.")
            return
        
        memberList = [member['name'] for member in self.groups[currentGroup]]
        combineMemberVocals(jsonFiles, vocalsOnlySongs, currentGroup, memberList)
        messagebox.showinfo("Done", "Combined and saved member vocals")
    
    def chooseSongForHarmonyExtraction(self):
        """
        Opens a scrollable song list and extracts harmonies + segments
        for training when a song is clicked.
        """
        selectedGroup = self.currentGroup.get()

        def onSongSelected(songName):
            print(f"Cry")

        self.chooseSongWindow(
            selectedGroup=selectedGroup,
            title=f"Extract Harmonies & Segments for {selectedGroup}",
            callback=onSongSelected
        )
    
    def extractSong(self):
        group = self.chooseGroup()
        if group == "Back": return
        vocalsPath = f"./training_data/{group}"
        vocalsOnly = [f for f in os.listdir(vocalsPath) if (f.endswith(".mp3") or f.endswith(".wav")) and "_vocals" in f]
        if not vocalsOnly:
            print("No songs available.")
            return
        labelDir = f"./saved_labels/{group}"
        labelFiles = [os.path.join(labelDir, f) for f in os.listdir(labelDir) if f.endswith(".json")]
        combineMemberVocals(labelFiles, vocalsOnly, group)
        
    def openAddGroupDialog(self):
        win = tk.Toplevel(self.root)
        win.title("Add New Group")
        win.transient(self.root)
        win.grab_set()
        
        # ---- Variables ----
        groupNameVar = tk.StringVar(value="")
        groupDirVar = tk.StringVar(value="")   # ./group_icons/<group>
        themeVar = tk.StringVar(value="")      # subfolder name
        
        # Default templates (your covention)
        darkTplVar = tk.StringVar(value="Dark {member}.png")
        lightTplVar = tk.StringVar(value="{member}.png")
        circleTplVar = tk.StringVar(value="{member} Circle.png")
        
        # --- Layout ---
        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        
        ttk.Label(frm, text="Group name (folder name):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=groupNameVar, width=30).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(frm, text="Group icons folder:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=groupDirVar).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        
        def chooseGroupFolder():
            # user selects ./group_icons/<group> folder
            selected = filedialog.askdirectory(title="Select the group folder (e.g. ./group_icons/aespa)")
            if selected:
                groupDirVar.set(selected)
                # Auto-fill groupName from folder name if empty
                if not groupNameVar.get().strip():
                    groupNameVar.set(Path(selected).name)
                refreshThemes()
        
        ttk.Button(frm, text="Browse...", command=chooseGroupFolder).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Theme subfolder (e.g. Whiplash):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        themeDropdown = ttk.Combobox(frm, textvariable=themeVar, values=[], state="readonly")
        themeDropdown.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        
        def refreshThemes():
            base = groupDirVar.get().strip()
            if not base:
                themeDropdown["values"] = []
                themeVar.set("")
                return
            basePath = Path(base)
            if not basePath.exists():
                themeDropdown["values"] = []
                themeVar.set("")
                return

            subfolders = sorted([p.name for p in basePath.iterdir() if p.is_dir()], key=str.lower)
            themeDropdown["values"] = subfolders
            if subfolders:
                themeVar.set(subfolders[0])
            else:
                themeVar.set("")

        ttk.Label(frm, text="Dark template (must include {member}):").grid(row=3, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(frm, textvariable=darkTplVar).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(12, 0))

        ttk.Label(frm, text="Light template (must include {member}):").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=lightTplVar).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Circle template (must include {member}):").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=circleTplVar).grid(row=5, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        statusVar = tk.StringVar(value="Define templates, then click Test.")
        statusLbl = ttk.Label(frm, textvariable=statusVar)
        statusLbl.grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 0))

        frm.columnconfigure(1, weight=1)
        
        # ---- Buttons ----
        btnRow = ttk.Frame(frm)
        btnRow.grid(row=7, column=0, columnspan=3, sticky="e", pady=(12, 0))

        def onTest():
            try:
                report = self._testGroupTemplates(
                    groupName=groupNameVar.get(),
                    groupDir=groupDirVar.get(),
                    theme=themeVar.get(),
                    darkTemplate=darkTplVar.get(),
                    lightTemplate=lightTplVar.get(),
                    circleTemplate=circleTplVar.get()
                )
                statusVar.set(report["summary"])
                if report["ok"]:
                    messagebox.showinfo("Templates OK", report["details"], parent=win)
                else:
                    messagebox.showwarning("Templates need fixes", report["details"], parent=win)
            except Exception as e:
                messagebox.showerror("Test failed", f"{type(e).__name__}: {e}", parent=win)
                
        def onSave():
            report = self._testGroupTemplates(
                groupName=groupNameVar.get(),
                groupDir=groupDirVar.get(),
                theme=themeVar.get(),
                darkTemplate=darkTplVar.get(),
                lightTemplate=lightTplVar.get(),
                circleTemplate=circleTplVar.get()
            )
            if not report["ok"]:
                messagebox.showwarning("Cannot Save", "Fix the issues shown in Test before saving.", parent=win)
                return
            
            try:
                self._saveGroupManifestAndUpdateUi(
                    groupName=report["groupName"],
                    groupDir=Path(report["groupDir"]),
                    theme=report["theme"],
                    templates=report["templates"],
                    members=report["members"]
                )
                messagebox.showinfo("Saved", f"Group '{report['groupName']}' added.", parent=win)
                win.destroy()
            except Exception as e:
                messagebox.showerror("Save failed", f"{type(e).__name__}: {e}", parent=win)
                
        ttk.Button(btnRow, text="Test", command=onTest).pack(side="right", padx=(8, 0))
        ttk.Button(btnRow, text="Save", command=onSave).pack(side="right", padx=(8, 0))
        ttk.Button(btnRow, text="Cancel", command=win.destroy).pack(side="right")
        
    def _testGroupTemplates(self, groupName, groupDir, theme, darkTemplate, lightTemplate, circleTemplate):
        groupName = (groupName or "").strip()
        groupDir = (groupDir or "").strip()
        theme = (theme or "").strip()

        if not groupName:
            return {"ok": False, "summary": "Missing group name.", "details": "Enter a group name.", "members": []}
        if not groupDir:
            return {"ok": False, "summary": "Missing group folder.", "details": "Choose the group folder.", "members": []}

        base = Path(groupDir)
        if not base.exists():
            return {"ok": False, "summary": "Group folder not found.", "details": f"Folder does not exist:\n{base}", "members": []}

        themeDir = base / theme if theme else None
        if themeDir is None or not themeDir.exists():
            return {"ok": False, "summary": "Theme folder missing.", "details": "Choose a theme subfolder (e.g. Whiplash).", "members": []}

        # Enforce consistent single-placeholder templates
        darkPrefix, darkSuffix = self._splitTemplate(darkTemplate.strip())
        lightPrefix, lightSuffix = self._splitTemplate(lightTemplate.strip())
        circlePrefix, circleSuffix = self._splitTemplate(circleTemplate.strip())

        # Infer members by matching dark template prefix/suffix against files
        files = [p.name for p in themeDir.iterdir() if p.is_file()]
        inferredMembers = []

        for fn in files:
            if fn.startswith(darkPrefix) and fn.endswith(darkSuffix):
                name = fn[len(darkPrefix):len(fn) - len(darkSuffix)]
                name = name.strip()
                if name:
                    inferredMembers.append(name)

        inferredMembers = sorted(set(inferredMembers), key=str.lower)
        if not inferredMembers:
            return {
                "ok": False,
                "summary": "No members detected from dark template.",
                "details": (
                    "I couldn't infer member names from your dark template.\n\n"
                    f"Dark template: {darkTemplate}\n"
                    f"Theme folder: {themeDir}\n\n"
                    "Rule: files must match <prefix>{member}<suffix> exactly."
                ),
                "members": []
            }

        # Validate required assets for every member
        missing = []
        membersOut = []

        for m in inferredMembers:
            darkPath = themeDir / (darkTemplate.format(member=m))
            lightPath = themeDir / (lightTemplate.format(member=m))
            circlePath = themeDir / (circleTemplate.format(member=m))

            if not darkPath.exists():
                missing.append(f"{m}: missing DARK -> {darkPath.name}")
            if not lightPath.exists():
                missing.append(f"{m}: missing BRIGHT -> {lightPath.name}")
            if not circlePath.exists():
                missing.append(f"{m}: missing CIRCLE -> {circlePath.name}")

            colorHex = ""
            if darkPath.exists():
                try:
                    colorHex = self.groupRegistry._inferHexFromImageCorner(darkPath)
                except Exception as e:
                    missing.append(f"{m}: failed color read from DARK ({darkPath.name}) -> {type(e).__name__}: {e}")

            membersOut.append({"name": m, "color": colorHex})

        ok = (len(missing) == 0)
        summary = f"Detected {len(inferredMembers)} members. " + ("All required files found." if ok else f"{len(missing)} issues found.")

        details = (
            f"Group: {groupName}\n"
            f"Theme: {theme}\n"
            f"Members detected: {', '.join(inferredMembers)}\n\n"
        )
        if missing:
            details += "Problems:\n" + "\n".join(missing)
        else:
            details += "Everything looks good. You can Save now."

        return {
            "ok": ok,
            "summary": summary,
            "details": details,
            "groupName": groupName,
            "groupDir": str(base),
            "theme": theme,
            "templates": {
                "dark": darkTemplate.strip(),
                "light": lightTemplate.strip(),
                "circle": circleTemplate.strip()
            },
            "members": membersOut
        }
        
    def _splitTemplate(self, template: str) -> tuple[str, str]:
        """
        Enforce template contains exactly one '{member}' placeholder.
        Returns (prefix, suffix) for filename matching. This should return an alert,
        not a console log 
        """
        if "{member}" not in template:
            raise ValueError(f"Template must contain {{member}}: {template}")

        parts = template.split("{member}")
        if len(parts) != 2:
            raise ValueError(f"Template must contain exactly one {{member}}: {template}")

        prefix, suffix = parts[0], parts[1]
        return prefix, suffix
    
    def _saveGroupManifestAndUpdateUi(self, groupName: str, groupDir: Path, theme: str, templates: dict, members: list[dict]):
        groupDir = Path(groupDir)
        manifestPath = groupDir / "group.json"

        manifest = {
            "group": groupName,
            "activeTheme": theme,
            "templates": templates,
            "members": members
        }

        with manifestPath.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        # Update in-memory groups so GUI sees it immediately
        self.groups[groupName]["members"] = members

        # Refresh dropdown if you already store it on self (recommended)
        if hasattr(self, "groupDropdown") and self.groupDropdown:
            groupNames = sorted(list(self.groups.keys()), key=str.lower)
            self.groupDropdown["values"] = groupNames

        # Switch current group to the new one and display members
        if hasattr(self, "currentGroup"):
            self.currentGroup.set(groupName)
        self.displayMembers(groupName)
              
if __name__ == "__main__":
    root = tk.Tk()
    
    icon = tk.PhotoImage(file="./images/logo.png")
    root.iconphoto(True, icon)
    
    app = VoiceTrainerGUI(root)
    root.mainloop()