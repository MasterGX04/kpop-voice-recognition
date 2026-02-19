import json, os, sys
from pathlib import Path
from PIL import Image

def getAppRoot() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

class GroupRegistry:
    """
    Owns all group metadata persistence and discovery.
    - groups.json stores the list of groups + members (global registry)
    - ./group_icons/{group}/group.json stores per-group manifest (optional but recommended)
    """
    
    def __init__(self, iconsRoot="group_icons", groupsJsonPath="groups.json", defaultGroups=None):
        baseDir = getAppRoot()
        self.iconsRoot = baseDir / iconsRoot 
        os.makedirs(self.iconsRoot, exist_ok=True)
        self.groupsJsonPath = baseDir / groupsJsonPath
        self.groups = {}

        loaded = self.loadGroupsFromJson()
        if not loaded and defaultGroups:
            # Bootstrap from whatever is currently hardcoded, then persist
            self.groups = defaultGroups
            self.saveGroupsToJson()

        didChange = False
        didChange |= self._migrateGroupsShape()
        didChange |= self._ensureAlbumFields()
        # print(f"Did the group change?: {didChange}")
        
        if didChange:
            self.saveGroupsToJson()
        
        # Always do discovery on startup
        self.scanForNewGroups()

        # Fill missing colors for any group/member that lacks a color
        self.fillMissingColorsFromIcons()
        
        didChange = self._ensureMemberImageFields()

        # Persist any updates
        self.saveGroupsToJson()

    # ---------- JSON persistence ----------
    def loadGroupsFromJson(self) -> bool:
        if not self.groupsJsonPath.exists():
            return False
        try:
            with self.groupsJsonPath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.groups = data
                return True
        except Exception as e:
            print(f"[GroupRegistry] Failed to load {self.groupsJsonPath}: {type(e).__name__}: {e}")
        return False
    
    def saveGroupsToJson(self) -> None:
        try:
            self.groupsJsonPath.parent.mkdir(parents=True, exist_ok=True)
            with self.groupsJsonPath.open("w", encoding="utf-8") as f:
                json.dump(self.groups, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[GroupRegistry] Failed to save {self.groupsJsonPath}: {type(e).__name__}: {e}")
            
    def addGroupToJson(self, groupName: str, members: list[dict]) -> None:
        """
        Adds/overwrites a group entry, then saves to groups.json.
        """
        self.groups[groupName]["members"] = members
        self.setCurrentGroup(groupName)
        self.saveGroupsToJson()
        
    def setCurrentGroup(self, groupFunction) -> None:
        self.setCurrentGroup = groupFunction
        
    def getGroupDir(self, groupName: str) -> Path:
        # 1) per-group override stored in groups.json
        g = self.groups.get(groupName, {})
        if isinstance(g, dict):
            iconsDir = g.get("iconsDir")
            if isinstance(iconsDir, str) and iconsDir.strip():
                p = Path(iconsDir).expanduser()
                # allow relative paths relative to the registry file location
                if not p.is_absolute():
                    p = self.groupsJsonPath.parent / p
                return p

        # 2) default behavior (current)
        return self.iconsRoot / str(groupName)
        
    def scanForNewGroups(self) -> None:
        """
        Scans ./group_icons for new groups not in groups.json.
        For each new group, creates ./group_icons/{group}/group.json (manifest)
        and adds group to groups.json.
        """
        if not self.iconsRoot.exists():
            print(f"[GroupRegistry] iconsRoot not found: {self.iconsRoot}")
            return
        
        for groupDir in sorted([p for p in self.iconsRoot.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            groupName = groupDir.name
            if groupName in self.groups:
                # If group exists but missing members/colors, manifest can help
                continue
            
            manifest = self._loadGroupManifest(groupDir)
            if manifest is None:
                print(f"No manifest exists, building from scratch")
                manifest = self._buildGroupManifestFromIcons(groupDir)
                if manifest:
                    self._saveGroupManifest(groupDir, manifest)
                    
            if manifest and "members" in manifest and isinstance(manifest["members"], list):
                if groupName not in self.groups or self.groups[groupName] is None:
                    self.groups[groupName] = {}
                self.groups[groupName]["members"] = manifest["members"]
                print(f"[GroupRegistry] Discovered new group '{groupName}' with {len(manifest['members'])} members.")
            else:
                print(f"[GroupRegistry] Found group folder '{groupName}', but couldn't infer members. (No manifest + no matching files)")
    
    def inferGroupDirAndThemeFromPath(self, selectedPath) -> tuple[Path, str]:
        """
        User may pick either:
          A) group folder:  .../group_icons/ARTMS
          B) album folder:  .../group_icons/ARTMS/DALL

        Returns:
          (groupDir, themeNameToUse)
        """
        p = Path(selectedPath).expanduser().resolve()

        # If they picked an album folder, its parent is the group folder
        # Detect "album folder" by: it has PNGs / and parent exists
        if p.is_dir():
            # If folder contains any pngs, assume it's an album/theme folder
            hasPng = any(x.is_file() and x.suffix.lower() == ".png" for x in p.iterdir())
            if hasPng and p.parent.is_dir():
                return (p.parent, p.name)

        # Otherwise assume it's the group folder
        return (p, "")
    
    def _pickDefaultThemeFolder(self, groupDir: Path) -> Path:
        """
        Deterministic 'last album' folder. Uses sorted order by folder name.
        """
        subfolders = sorted([p for p in groupDir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
        if not subfolders:
            return None
        return subfolders[-1]  # <--- LAST instead of FIRST
    
    def _buildGroupManifestFromIcons(self, groupDir: Path, forcedTheme: str = ""):
        """
        Convention-based fallback:
        - Default theme: LAST subfolder (stable sorted), unless forcedTheme provided.
        - Detect members from Dark*.png inside theme folder.
        """
        if not groupDir.exists() or not groupDir.is_dir():
            return None

        if forcedTheme:
            themeDir = groupDir / forcedTheme
            if not themeDir.exists() or not themeDir.is_dir():
                raise FileNotFoundError(f"Theme folder not found: {themeDir}")
        else:
            themeDir = self._pickDefaultThemeFolder(groupDir)
            if themeDir is None:
                return None

        darkPngs = [p for p in themeDir.glob("Dark*.png") if p.is_file()]
        if not darkPngs:
            return None

        members = []
        for p in sorted(darkPngs, key=lambda x: x.name.lower()):
            memberName = self._parseMemberNameFromDarkFilename(p.name)
            if not memberName:
                continue

            colorHex = self._inferHexFromImageCorner(p)
            members.append({"name": memberName, "color": colorHex})

        if not members:
            return None

        return {
            "group": groupDir.name,
            "activeTheme": themeDir.name,
            "members": members,
            "ageOrder": [m["name"] for m in members],
            "templates": {
                "light": "{member}.png",
                "dark": "Dark {member}.png",
                "circle": "{member} Circle.png"
            }
        }
        
    def ensureManifestFromSelectedIconsPath(self, selectedPath) -> dict:
        """
        Ensures ./group_icons/{group}/group.json exists, and sets activeTheme properly:
        - If user selected .../{group}/{album}, activeTheme = album
        - Else activeTheme = LAST subfolder in .../{group}
        Returns the manifest dict.
        """
        groupDir, forcedTheme = self.inferGroupDirAndThemeFromPath(selectedPath)
        os.makedirs(groupDir, exist_ok=True)

        manifest = self._loadGroupManifest(groupDir) or {"group": groupDir.name}

        # If no manifest OR missing fields, build/refresh from icons
        built = self._buildGroupManifestFromIcons(groupDir, forcedTheme=forcedTheme) if forcedTheme else self._buildGroupManifestFromIcons(groupDir)

        if not built:
            raise ValueError(
                "Could not infer members/images from the selected folder.\n"
                "Make sure it contains Dark*.png files in the theme folder."
            )

        # Preserve any existing fields you care about, but ensure activeTheme matches our rule
        manifest["group"] = built["group"]
        manifest["activeTheme"] = built["activeTheme"]
        manifest["members"] = built["members"]
        manifest["ageOrder"] = built.get("ageOrder", [m["name"] for m in built["members"]])
        manifest["templates"] = manifest.get("templates") or built["templates"]

        self._saveGroupManifest(groupDir, manifest)

        # Also keep runtime registry consistent
        if groupDir.name not in self.groups or self.groups[groupDir.name] is None:
            self.groups[groupDir.name] = {"members": [], "albums": {}, "songToAlbum": {}, "defaultAlbumId": ""}
        self.groups[groupDir.name]["members"] = manifest["members"]
        self.saveGroupsToJson()

        return manifest
    
    # ---------- group.json manifest helpers ----------              
    def _loadGroupManifest(self, groupDir: Path):
        manifestPath = groupDir / "group.json"
        if not manifestPath.exists():
            return None
        try:
            with manifestPath.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return None

            # --- HARDEN ageOrder ---
            members = data.get("members", [])
            if isinstance(members, list):
                memberNames = [
                    (m.get("name") or "").strip()
                    for m in members
                    if isinstance(m, dict) and isinstance(m.get("name"), str)
                ]
                memberNames = [n for n in memberNames if n]

                ageOrder = data.get("ageOrder")

                # Missing / invalid / wrong length → default alphabetical
                if (
                    not isinstance(ageOrder, list)
                    or len(ageOrder) != len(memberNames)
                    or set(ageOrder) != set(memberNames)
                ):
                    defaultOrder = sorted(memberNames, key=str.lower)
                    data["ageOrder"] = defaultOrder

                    print(
                        f"[GroupRegistry] '{groupDir.name}': "
                        f"Missing or invalid ageOrder → defaulting to alphabetical order"
                    )

            return data

        except Exception as e:
            print(f"[GroupRegistry] Failed to read manifest {manifestPath}: {type(e).__name__}: {e}")
            return None
    
    def getGroupMediaDir(self, groupName: str) -> str:
        """
        Returns the media directory for a group.
        If not set in group.json, defaults to ./training_data/{groupName}.
        """
        groupDir = self.getGroupDir(groupName)
        manifest = self._loadGroupManifest(groupDir) or {}
        mediaDir = manifest.get("mediaDir")

        if isinstance(mediaDir, str) and mediaDir.strip():
            return mediaDir.strip()

        return os.path.join(".", "training_data", groupName)
    
    def setGroupMediaDir(self, groupName: str, mediaDir: str) -> None:
        """
        Persists a group's media directory into ./group_icons/{groupName}/group.json
        """
        groupDir = self.getGroupDir(groupName)
        manifestPath = os.path.join(groupDir, "group.json")

        manifest = self._loadGroupManifest(groupDir) or {}

        # Normalize path (optional, but nice)
        manifest["mediaDir"] = os.path.normpath(mediaDir)

        os.makedirs(groupDir, exist_ok=True)
        with open(manifestPath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
    
    def _saveGroupManifest(self, groupDir: Path, manifest: dict) -> None:
        manifestPath = groupDir / "group.json"
        try:
            with manifestPath.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[GroupRegistry] Failed to write manifest {manifestPath}: {type(e).__name__}: {e}")
            
    def _buildGroupManifestFromIcons(self, groupDir: Path):
        """
        Convention-based fallback:
        - Use the first subfolder as the 'theme' (stable sorted).
        - Detect members by finding files matching: "Dark {name}.png" OR "Dark_{name}.png"
        - Infer member color from first non-transparent pixel near top-left.
        """
        subfolders = sorted([p for p in groupDir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
        if not subfolders:
            return None

        themeDir = subfolders[0]

        darkPngs = [p for p in themeDir.glob("Dark*.png") if p.is_file()]
        if not darkPngs:
            return None

        members = []
        for p in sorted(darkPngs, key=lambda x: x.name.lower()):
            memberName = self._parseMemberNameFromDarkFilename(p.name)
            if not memberName:
                continue

            colorHex = self._inferHexFromImageCorner(p)

            members.append({
                "name": memberName,
                "color": colorHex
            })

        if not members:
            return None

        return {
            "group": groupDir.name,
            "activeTheme": themeDir.name,
            "members": members,
            "ageOrder": [m["name"] for m in members], # Default dispaly order = scan order
            "templates": {
                "light": "{member}.png",
                "dark": "Dark {member}.png",
                "circle": "{member} Circle.png"
            }
        }
        
    def _parseMemberNameFromDarkFilename(self, filename: str):
        # Accept: "Dark Ningning.png" and "Dark_Ningning.png"
        base = filename[:-4] if filename.lower().endswith(".png") else filename
        if not base.lower().startswith("dark"):
            return None

        # Strip "Dark" prefix + separators
        rest = base[4:]  # after "Dark"
        rest = rest.lstrip(" _-")  # allow "Dark ", "Dark_", "Dark-"
        rest = rest.strip()
        return rest if rest else None
    
    def _ensureMemberImageFields(self) -> bool:
        """
        Ensures every member dict has the keys used by the GUI for custom images.
        Returns True if anything changed (so caller can save).
        """
        changed = False

        for groupName, members in (self.groups or {}).items():
            if not isinstance(members, list):
                continue

            for m in members:
                if not isinstance(m, dict):
                    continue

                # Add new fields with safe defaults if missing
                if "imageCachePath" not in m:
                    m["imageCachePath"] = ""
                    changed = True

                # Optional: only if you’re supporting URL input too
                if "imageUrl" not in m:
                    m["imageUrl"] = ""
                    changed = True

        return changed
    
    # ---------- Color filling ----------
    def fillMissingColorsFromIcons(self) -> dict:
        """
        For each group/member in `groups`, if member['color'] is missing/empty,
        infer it from the (0,0) pixel of:
        ./group_icons/{group}/{first_subfolder}/Dark_{memberName}.png

        Mutates and also returns `groups`.
        """
        for groupName, members in self.groups.items():
            if not isinstance(members, list) or not members:
                continue

            groupDir = self.iconsRoot / str(groupName)
            if not groupDir.exists():
                continue
            
            manifest = self._loadGroupManifest(groupDir)
            themeDir = None
            if manifest and isinstance(manifest.get("activeTheme"), str):
                candidate = groupDir / manifest["activeTheme"]
                if candidate.exists() and candidate.is_dir():
                    themeDir = candidate

            if themeDir is None:
                subfolders = sorted([p for p in groupDir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
                if not subfolders:
                    continue
                themeDir = subfolders[0]

            for member in members:
                if not isinstance(member, dict):
                    continue
                if isinstance(member.get("color"), str) and member["color"].strip():
                    continue

                name = member.get("name", "")
                if not isinstance(name, str) or not name.strip():
                    continue

                # Accept both "Dark {name}.png" and "Dark_{name}.png"
                p1 = themeDir / f"Dark {name}.png"
                p2 = themeDir / f"Dark_{name}.png"
                imagePath = p1 if p1.exists() else p2

                if not imagePath.exists():
                    print(f"[Color Inference] Missing inactive image for {groupName}/{name}: {p1} OR {p2}")
                    continue

                try:
                    member["color"] = self._inferHexFromImageCorner(imagePath)
                except Exception as e:
                    print(
                        "[Color Inference ERROR]\n"
                        f"  Group: {groupName}\n"
                        f"  Member: {name}\n"
                        f"  Image path: {imagePath}\n"
                        f"  Error type: {type(e).__name__}\n"
                        f"  Message: {e}\n"
                    )
                    
    def _inferHexFromImageCorner(self, imagePath: Path) -> str:
        """
        Reads pixel color near (0,0). If fully transparent, scans a small area
        until it finds a non-transparent pixel.
        """
        with Image.open(imagePath) as img:
            img = img.convert("RGBA")

            # scan a small top-left region for first non-transparent pixel
            scanSize = 12
            w, h = img.size
            maxX = min(scanSize, w)
            maxY = min(scanSize, h)

            for y in range(maxY):
                for x in range(maxX):
                    r, g, b, a = img.getpixel((x, y))
                    if a != 0:
                        return f"#{r:02x}{g:02x}{b:02x}"

            # If everything is transparent in region, fall back to literal (0,0)
            r, g, b, a = img.getpixel((0, 0))
            return f"#{r:02x}{g:02x}{b:02x}"
        
    def _normalizeName(self, name: str) -> str:
        return (name or "").strip()
    
    def _parseAgeCsv(self, ageCsv: str) -> list[str]:
        parts=[p.strip() for p in (ageCsv or "").split(",")]
        return [p for p in parts if p]
    
    def _validateOrderListOrThrow(self, groupName: str, members: list[dict], orderList: list[str]) -> None:
        memberNames = [self._normalizeName(m.get("name", "")) for m in members if isinstance(m, dict)]
        memberNames = [n for n in memberNames if n]
        
        memberSet = set(memberNames)
        orderSet = set(orderList)
        
        if len(orderList) != len(orderSet):
            seen = set()
            dups = []
            for n in orderList:
                if n in seen and n not in dups:
                    dups.append(n)
                seen.add(n)
            raise ValueError(f"[Order ERROR] '{groupName}': duplicate name(s): {', '.join(dups)}")
        
        missing = [n for n in memberNames if n not in orderSet]
        extra = [n for n in orderList if n not in memberSet]

        if missing or extra:
            msg = f"[Order ERROR] '{groupName}' order mismatch.\n"
            if missing:
                msg += f"  Missing member(s): {', '.join(missing)}\n"
            if extra:
                msg += f"  Unknown/extra name(s): {', '.join(extra)}\n"
            msg += "  Fix spelling and include every member exactly once."
            raise ValueError(msg)
        
    def _applyOrderToMembers(self, members: list[dict], orderList: list[str]) -> list[dict]:
        lookup = {self._normalizeName(m.get("name", "")): m for m in members if isinstance(m, dict)}
        return [lookup[name] for name in orderList]
    
    def _splitTemplate(self, template: str) -> tuple[str, str]:
        if "{member}" not in template:
            raise ValueError(f"Template must contain {{member}}: {template}")
        parts = template.split("{member}")
        if len(parts) != 2:
            raise ValueError(f"Template must contain exactly one {{member}}: {template}")
        return parts[0], parts[1]
    
    def _inferMembersFromDarkTemplate(self, themeDir: Path, darkTemplate: str) -> list[str]:
        prefix, suffix = self._splitTemplate(darkTemplate)
        inferred = []
        for p in themeDir.iterdir():
            if not p.is_file():
                continue
            fn = p.name
            if fn.startswith(prefix) and fn.endswith(suffix):
                name = fn[len(prefix):len(fn) - len(suffix)].strip()
                if name:
                    inferred.append(name)
        # Keep deterministic order but NOT alphabetical by default; keep filesystem order if possible.
        # Path.iterdir() order is OS-dependent; we’ll stabilize by filename sort for *scan*,
        # but you can override display order later via ageOrder.
        return sorted(set(inferred), key=str.lower)
    
    def ensureAgeOrder(self, groupName: str) -> None:
        """
        Ensure group.json has 'ageOrder' (display order).
        Default is current members order (scan order / existing order).
        Then reorder members to match and persist.
        """
        groupDir = self.iconsRoot / str(groupName)
        manifest = self._loadGroupManifest(groupDir)
        if not manifest:
            return

        members = manifest.get("members", [])
        if not isinstance(members, list) or not members:
            return

        ageOrder = manifest.get("ageOrder")
        if not isinstance(ageOrder, list) or not ageOrder:
            ageOrder = [self._normalizeName(m.get("name", "")) for m in members if isinstance(m, dict)]
            manifest["ageOrder"] = ageOrder

        # strict validation (if user edited it and made a mistake, error loudly)
        self._validateOrderListOrThrow(groupName, members, ageOrder)

        orderedMembers = self._applyOrderToMembers(members, ageOrder)
        manifest["members"] = orderedMembers

        self._saveGroupManifest(groupDir, manifest)

        # keep runtime + groups.json consistent
        self.groups[groupName]["members"] = orderedMembers
        
    def setAgeOrderFromCsv(self, groupName: str, ageCsv: str) -> None:
        """
        User-defined display order. Strict.
        Example: "Gaeul,Yujin,Rei,Wonyoung,Liz,Leeseo"
        """
        groupName = (groupName or "").strip()
        if not groupName:
            raise ValueError("[Order ERROR] Missing groupName.")

        groupDir = self.iconsRoot / groupName
        manifest = self._loadGroupManifest(groupDir)
        if not manifest:
            raise FileNotFoundError(f"[Order ERROR] group.json not found for '{groupName}'")

        members = manifest.get("members", [])
        if not isinstance(members, list) or not members:
            raise ValueError(f"[Order ERROR] group '{groupName}' has no members in manifest.")

        ageOrder = self._parseAgeCsv(ageCsv)
        if not ageOrder:
            raise ValueError("[Order ERROR] Empty order string. Provide comma-separated names.")

        self._validateOrderListOrThrow(groupName, members, ageOrder)

        orderedMembers = self._applyOrderToMembers(members, ageOrder)
        manifest["ageOrder"] = ageOrder
        manifest["members"] = orderedMembers

        self._saveGroupManifest(groupDir, manifest)
        self.groups[groupName]["members"] = orderedMembers
        self.saveGroupsToJson()
        
    def testEditGroupOrThrow(self, groupName: str, activeTheme: str, templates: dict, ageCsv: str) -> dict:
        """
        Validates + builds the updated member list. Throws on any failure.
        Returns a report dict including updatedMembers + ageOrder list.
        """
        groupName = (groupName or "").strip()
        if not groupName:
            raise ValueError("Missing group name.")

        groupDir = self.getGroupDir(groupName)
        if not groupDir.exists():
            raise FileNotFoundError(f"Group folder not found: {groupDir}")

        themeDir = groupDir / (activeTheme or "")
        if not themeDir.exists():
            raise FileNotFoundError(f"Theme folder not found: {themeDir}")
        
        # Templates required keys
        for k in ("dark", "light", "circle"):
            if k not in templates or not isinstance(templates[k], str):
                raise ValueError(f"Missing template '{k}'")
            
        darkT = templates["dark"].strip()
        lightT = templates["light"].strip()
        circleT = templates["circle"].strip()
        
        # Ensure placeholders exist
        self._splitTemplate(darkT)
        self._splitTemplate(lightT)
        self._splitTemplate(circleT)
        
        members = self._inferMembersFromDarkTemplate(themeDir, darkT)
        
        if not members:
            raise ValueError(
                "No members detected from dark template.\n"
                f"Dark template: {darkT}\n"
                f"Theme folder: {themeDir}"
            )
            
        missing = []
        membersOut = []
        
        for m in members:
            darkPath = themeDir / darkT.format(member=m)
            lightPath = themeDir / lightT.format(member=m)
            circlePath = themeDir / circleT.format(member=m)

            if not darkPath.exists():
                missing.append(f"{m}: missing DARK -> {darkPath.name}")
            if not lightPath.exists():
                missing.append(f"{m}: missing BRIGHT -> {lightPath.name}")
            if not circlePath.exists():
                missing.append(f"{m}: missing CIRCLE -> {circlePath.name}")

            colorHex = ""
            if darkPath.exists():
                colorHex = self._inferHexFromImageCorner(darkPath)

            membersOut.append({"name": m, "color": colorHex})

        if missing:
            raise ValueError("Template check failed:\n" + "\n".join(missing))
        
        # Age/display order
        if ageCsv is not None and ageCsv.strip():
            ageOrder = self._parseAgeCsv(ageCsv)
            self._validateOrderListOrThrow(groupName, membersOut, ageOrder)
        else:
            # Keep existing ageOrder if it matches; else default to scan order
            manifest = self._loadGroupManifest(groupDir) or {}
            existing = manifest.get("ageOrder")
            if isinstance(existing, list) and existing:
                self._validateOrderListOrThrow(groupName, membersOut, existing)
                ageOrder = existing
            else:
                ageOrder = [m["name"] for m in membersOut]

        # Apply order to members
        orderedMembers = self._applyOrderToMembers(membersOut, ageOrder)

        return {
            "groupName": groupName,
            "groupDir": str(groupDir),
            "theme": activeTheme,
            "templates": {"dark": darkT, "light": lightT, "circle": circleT},
            "members": orderedMembers,
            "ageOrder": ageOrder
        }
        
    def saveEditGroup(self, groupName: str, activeTheme: str, templates: dict, ageCsv: str) -> None:
        """
        Performs test + writes group.json + updates self.groups + groups.json.
        """
        report = self.testEditGroupOrThrow(groupName, activeTheme, templates, ageCsv)

        groupDir = self.getGroupDir(groupName)
        manifest = self._loadGroupManifest(groupDir) or {"group": groupName}

        manifest["activeTheme"] = report["theme"]
        manifest["templates"] = report["templates"]
        manifest["members"] = report["members"]
        manifest["ageOrder"] = report["ageOrder"]

        self._saveGroupManifest(groupDir, manifest)

        # Update runtime registry + groups.json
        self.groups[groupName]["members"] = report["members"]
        self.saveGroupsToJson()
        
    def _migrateGroupsShape(self) -> bool:
        """
        Migrate self.groups into a normalized shape:
        self.groups[groupName] = {
            "members": [...],
            "albums": {...},
            "songToAlbum": {...},
            "defaultAlbumId": str
        }
        Returns True if anything changed.
        """
        changed = False
        newGroups = {}

        for groupName, groupVal in (self.groups or {}).items():
            # Old shape: groupVal is a list of members
            if isinstance(groupVal, list):
                newGroups[groupName] = {
                    "members": groupVal,
                    "albums": {},
                    "songToAlbum": {},
                    "defaultAlbumId": ""
                }
                changed = True
                continue

            # New shape: dict with members/albums
            if isinstance(groupVal, dict):
                members = groupVal.get("members", [])
                albums = groupVal.get("albums", {})
                songToAlbum = groupVal.get("songToAlbum", {})
                defaultAlbumId = groupVal.get("defaultAlbumId", "")

                if "members" not in groupVal:
                    groupVal["members"] = members
                    changed = True
                if "albums" not in groupVal:
                    groupVal["albums"] = albums
                    changed = True
                if "songToAlbum" not in groupVal:
                    groupVal["songToAlbum"] = songToAlbum
                    changed = True
                if "defaultAlbumId" not in groupVal:
                    groupVal["defaultAlbumId"] = defaultAlbumId
                    changed = True

                newGroups[groupName] = groupVal
                continue

            # Unexpected; normalize to empty
            newGroups[groupName] = {
                "members": [],
                "albums": {},
                "songToAlbum": {},
                "defaultAlbumId": ""
            }
            changed = True

        self.groups = newGroups
        return changed
    
    def _ensureAlbumFields(self) -> bool:
        """
        Ensure album metadata keys exist and are well-formed.
        Returns True if changed.
        """
        changed = False
        for groupName, g in (self.groups or {}).items():
            if not isinstance(g, dict):
                continue

            albums = g.get("albums", {})
            if not isinstance(albums, dict):
                g["albums"] = {}
                albums = g["albums"]
                changed = True

            # Ensure each album has art fields + song list
            for albumId, album in list(albums.items()):
                if not isinstance(album, dict):
                    albums[albumId] = {"displayName": str(albumId), "albumArtCachePath": "", "albumArtUrl": "", "songs": []}
                    changed = True
                    continue

                if "displayName" not in album:
                    album["displayName"] = str(albumId)
                    changed = True
                if "albumArtCachePath" not in album:
                    album["albumArtCachePath"] = ""
                    changed = True
                if "albumArtUrl" not in album:
                    album["albumArtUrl"] = ""
                    changed = True
                if "songs" not in album or not isinstance(album.get("songs"), list):
                    album["songs"] = []
                    changed = True

            # Ensure songToAlbum exists
            if "songToAlbum" not in g or not isinstance(g.get("songToAlbum"), dict):
                g["songToAlbum"] = {}
                changed = True

            if "defaultAlbumId" not in g:
                g["defaultAlbumId"] = ""
                changed = True

        return changed

    def getMembers(self, groupName):
        g = self.groups.get(groupName)
        if isinstance(g, dict):
            return g.get("members", [])
        return g or []

    def getAlbums(self, groupName):
        g = self.groups.get(groupName, {})
        if isinstance(g, dict):
            return g.get("albums", {})
        return {}
    
    def setSongAlbum(self, groupName, songName, albumId):
        g = self.groups[groupName]
        if "albums" not in g:
            self._ensureAlbumFields()
        albums = g["albums"]
        if albumId not in albums:
            raise ValueError(f"Album '{albumId}' not found in group '{groupName}'")

        # Remove song from previous album list if present
        prev = g["songToAlbum"].get(songName)
        if prev and prev in albums:
            if songName in albums[prev].get("songs", []):
                albums[prev]["songs"].remove(songName)

        # Assign new
        g["songToAlbum"][songName] = albumId
        if songName not in albums[albumId]["songs"]:
            albums[albumId]["songs"].append(songName)

        self.saveGroupsToJson()
        
    def getSongAlbumId(self, groupName, songName):
        g = self.groups.get(groupName, {})
        if not isinstance(g, dict):
            return ""
        albumId = g.get("songToAlbum", {}).get(songName, "")
        if albumId:
            return albumId
        return g.get("defaultAlbumId", "") or ""
    
    def createAlbum(self, groupName, albumId, displayName=None):
        g = self.groups[groupName]
        if "albums" not in g:
            self._ensureAlbumFields()
            
        albums = g["albums"]

        if albumId in albums:
            raise ValueError(f"Album already exists: {albumId}")

        albums[albumId] = {
            "displayName": displayName or albumId,
            "albumArtCachePath": "",
            "albumArtUrl": "",
            "songs": []
        }

        if not g["defaultAlbumId"]:
            g["defaultAlbumId"] = albumId

        self.saveGroupsToJson()
        
    def setAlbumArt(self, groupName, albumId, cachePath="", url=""):
        albums = self.groups[groupName]["albums"]
        if albumId not in albums:
            raise ValueError(f"Album not found: {albumId}")

        albums[albumId]["albumArtCachePath"] = cachePath or ""
        albums[albumId]["albumArtUrl"] = url or ""
        self.saveGroupsToJson()
        
    def loadMemberImages(self, groupName: str, groupManifest: dict, songName: str):
        """
        Returns (albumNameUsed, memberImages)

        albumNameUsed:
            - song's albumId if present
            - else groupManifest["activeTheme"]

        memberImages structure:
            {
              "Jin": {"dark": PIL.Image, "light": PIL.Image, "circle": PIL.Image},
              ...
            }

        Uses:
            - groupManifest["templates"] for filename templates (keys are image types)
            - groupManifest["ageOrder"] to decide which members to load and in what order
        """
        activeTheme = (groupManifest.get("activeTheme", "") or "").strip()
        templates = groupManifest.get("templates", {}) or {}
        ageOrder = groupManifest.get("ageOrder", []) or []
        
        groupDir = self.getGroupDir(groupName)
        groupBaseDir = str(groupDir)
        # 1) Decide theme folder: song album first, else activeTheme fallback
        def dirExists(themeName: str) -> bool:
            return bool(themeName) and os.path.isdir(os.path.join(groupBaseDir, themeName))
        
        albumId = ""
        try:
            albumId = self.getSongAlbumId(groupName, songName) or ""
        except Exception:
            albumId = ""
        
        albumTheme = ""
        if albumId:
            try:
                albums = self.getAlbums(groupName)  # safer than self.groups[...] direct
                albumTheme = (albums.get(albumId, {}).get("displayName", "") or "").strip()
            except Exception:
                print("Can't find albumTheme!")
                albumTheme = ""
            
        # Prefer albumTheme IF its folder exists, otherwise fall back to activeTheme IF its folder exists
        primaryTheme = albumTheme if dirExists(albumTheme) else (activeTheme if dirExists(activeTheme) else "")
        
        if not os.path.isdir(groupBaseDir):
            print(f"[⚠️] Group icons folder missing: {groupBaseDir}. Using empty images.")
            memberImages = {m: {k: None for k in templates.keys()} for m in ageOrder}
            return (activeTheme or albumTheme or ""), memberImages
 
        memberImages = {}
        for memberName in ageOrder:
            perMember = {}

            for templateKey, filenameTemplate in templates.items():
                filename = filenameTemplate.format(member=memberName)
                # Try primary theme first
                candidatePaths = []
                if primaryTheme:
                    candidatePaths.append(os.path.join(groupBaseDir, primaryTheme, filename))

                # Fallback: activeTheme, but only if it’s different and exists
                if activeTheme and activeTheme != primaryTheme and dirExists(activeTheme):
                    candidatePaths.append(os.path.join(groupBaseDir, activeTheme, filename))

                loaded = None
                for imgPath in candidatePaths:
                    if os.path.exists(imgPath):
                        try:
                            loaded = Image.open(imgPath)
                            break
                        except Exception as e:
                            print(f"[⚠️] Failed to open image: {imgPath} ({e})")

                if loaded is None:
                    # Helpful log, but not noisy if you prefer — you can remove this print later
                    if candidatePaths:
                        print(f"[⚠️] Missing image for '{memberName}' ({templateKey}). Tried: {candidatePaths}")
                    else:
                        print(f"[⚠️] No valid theme folders available for '{groupName}'.")
                perMember[templateKey] = loaded

            memberImages[memberName] = perMember

        # Return the folder we actually used (for the UI to know theme),
        # but if none existed, return activeTheme as the "intent".
        themeUsed = primaryTheme or activeTheme or albumTheme or ""
        return memberImages