import tkinter as tk
from tkinter import ttk, messagebox
import os
from PIL import Image, ImageTk
from audio_processing import combineMemberVocals, getSongsFromSameAlbum, extractAndSaveHarmoniesFromSong
from audio_tester import VoiceDetectionApp, loadMemberImages
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from voice_classifier import train

class VoiceTrainerGUI:
    def __init__(self, root, groups):
        self.root = root
        self.groups = groups
        self.currentGroup = tk.StringVar(value=list(groups.keys())[0])
        self.imageSize = (100, 100)
        self.memberPhotos = {}
        self.memberImageRefs = []
        self.initPlaceholders()
        
        self.root.title("Voice Trainer")
        self.root.geometry("900x700")
        
        self.createWidgets()
        self.displayMembers(self.currentGroup.get())
        
    def createWidgets(self):
        # Topframe: group select
        topFrame = tk.Frame(self.root)
        topFrame.pack(pady=10)
        
        groupLabel = tk.Label(topFrame, text="Choose a K-pop group:")
        groupLabel.pack(side=tk.LEFT, padx=5)
        
        groupDropdown = ttk.Combobox(topFrame, textvariable=self.currentGroup, values=list(self.groups.keys()), state="readonly")
        groupDropdown.pack(side=tk.LEFT)
        groupDropdown.bind("<<ComboboxSelected>>", lambda e: self.displayMembers(self.currentGroup.get()))
        
        # Center: Scrollable vertical list
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container)
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
        
        tk.Button(bottomFrame, text="Train", command=self.trainModel).pack(side=tk.LEFT, padx=10)
        tk.Button(bottomFrame, text="Test", command=self.testModel).pack(side=tk.LEFT, padx=10)
        tk.Button(bottomFrame, text="Get Vocals Chart", command=self.visualizeMemberVocals).pack(side=tk.LEFT, padx=10)
        tk.Button(bottomFrame, text="Extract Member Vocals", command=self.combineAllVocalsFromGroup).pack(side=tk.LEFT, padx=10)
        tk.Button(
            bottomFrame,
            text="Extract Harmonies",
            command=self.chooseSongForHarmonyExtraction,
            font=("Helvetica", 12)
        ).pack(side=tk.LEFT, padx=10)
        
    def displayMembers(self, groupName):
        for widget in self.scrollFrame.winfo_children():
            widget.destroy()
            
        self.memberImageRefs.clear()
        
        members = self.groups[groupName]
        
        for member in members:
            memberName = member['name']
            frame = tk.Frame(self.scrollFrame, pady=5)
            frame.pack(fill="x", padx=0)

            image = self.loadMemberImage(groupName, member)
            self.memberImageRefs.append(image)
            
            labelImage = tk.Label(frame, image=image)
            labelImage.pack(side="left")

            labelText = tk.Label(frame, text=memberName, font=("Helvetica", 18), fg="black")
            labelText.pack(side="left", padx=20)
         
    def loadMemberImage(self, groupName, member):
        imagePath = os.path.join(groupName, "images", f"{member['name']}.png")
        try:
            image = Image.open(imagePath).resize(self.imageSize)
        except Exception:
            image = Image.new("RGB", self.imageSize, color=member['color'])
        
        return ImageTk.PhotoImage(image)
        
    def trainModel(self):
        selectedGroup = self.currentGroup.get()
        memberList = [member['name'] for member in self.groups[selectedGroup]]
        print(f"Starting to train on vocals forr {selectedGroup}")
        train(selectedGroup, memberList)
    
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
        
    def getAlbumForSong(self, group, songName):
        songAlbums = getSongsFromSameAlbum()
        groupAlbums = songAlbums.get(group, {})
        
        for albumName, songs in groupAlbums.items():
            if songName in songs:
                return albumName
        
        return None
    
    def initPlaceholders(self):
        try:
            img = Image.open("placeholder.png").resize((100, 100))
        except Exception as e:
            print(f"[⚠️] Failed to load placeholder.png: {e}")
            img = Image.new("RGB", (100, 100), "gray")

        self.placeholderIcon = ImageTk.PhotoImage(img)
    
    def chooseSongWindow(self, selectedGroup, title, callback):
        """
        Opens a scrollable window with all available songs in the group.
        When a song is clicked, the callback(songName) is triggered.
        """
        songDir = f"./training_data/{selectedGroup}"
        songList = [f.replace(".mp3", "") for f in os.listdir(songDir) if f.endswith(".mp3") and "_vocals" not in f]

        if not songList:
            print("❌ No songs available.")
            return

        self.albumImageRefs = []

        songWindow = tk.Toplevel(self.root)
        songWindow.title(title)
        songWindow.geometry("900x700")

        canvas = tk.Canvas(songWindow)
        frame = tk.Frame(canvas)
        scrollbar = tk.Scrollbar(songWindow, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.enableGlobalScroll(canvas, songWindow)

        for songName in songList:
            songFrame = tk.Frame(frame, pady=0)
            songFrame.pack(fill="x", padx=5)

            albumName = self.getAlbumForSong(selectedGroup, songName)
            if albumName:
                albumImagePath = os.path.join(selectedGroup, "images", f"{albumName}.png")
                try:
                    image = Image.open(albumImagePath).resize((100, 100))
                    songIcon = ImageTk.PhotoImage(image)
                except Exception:
                    print(f"[⚠️] Couldn't load {albumImagePath}, using placeholder.")
                    songIcon = self.placeholderIcon
            else:
                songIcon = self.placeholderIcon

            self.albumImageRefs.append(songIcon)

            imgLabel = tk.Label(songFrame, image=songIcon)
            imgLabel.pack(side="left", padx=5)

            button = tk.Button(
                songFrame,
                text=songName,
                font=("Helvetica", 14),
                anchor="w",
                justify="left",
                command=lambda name=songName: [songWindow.destroy(), callback(name)]
            )
            button.pack(side="left", fill="x", expand=True)
        
    def testModel(self):
        selectedGroup = self.currentGroup.get()
        songDir = f"./training_data/{selectedGroup}"
        modelPath = f"./models/{selectedGroup}_ecapa_head.pt"

        def launchVoiceApp(songName):
            testSongPath = os.path.join(songDir, f"{songName}.mp3")
            vocalsOnlyPath = os.path.join(songDir, f"{songName}_vocals.mp3")
            
            if not os.path.exists(vocalsOnlyPath):
                print(f"⚠️ Vocals-only file not found for {songName}")
                vocalsOnlyPath = testSongPath  # fallback

            images = loadMemberImages(selectedGroup, self.groups[selectedGroup], testSongPath)
            appWindow = tk.Toplevel(self.root)
            continueApp = [True]
            memberList = self.groups[selectedGroup][0]['name']
            app = None

            def onClose():
                if tk.messagebox.askyesno("Exit", "Do you want to stop the application?"):
                    if hasattr(app, "videoTrackItem") and app.videoTrackItem:
                        app.videoTrackItem.pause()
                        app.videoTrackItem.stop()
                    continueApp[0] = False
                    root.destroy()
                    sys.exit()
                else:
                    if tk.messagebox.askyesno("Switch Member/Group", "Do you want to switch to a different member or group?"):
                        if hasattr(app, "videoTrackItem") and app.videoTrackItem:
                            app.videoTrackItem.stop()
                        continueApp[0] = False
                        root.destroy()

            root.protocol("WM_DELETE_WINDOW", onClose)

            app = VoiceDetectionApp(
                appWindow,
                memberList,
                self.groups[selectedGroup],
                modelPath,
                images,
                testSongPath,
                vocalsOnlyPath,
                selectedGroup
            )
            if hasattr(app, "videoTrackItem") and app.videoTrackItem and app.videoTrackItem.thread:
                app.videoTrackItem.thread.daemon = True

            root.mainloop()

        # ✅ Launch song picker
        self.chooseSongWindow(
            selectedGroup=selectedGroup,
            title=f"Choose a song for {selectedGroup}",
            callback=launchVoiceApp
        )
    
    def visualizeMemberVocals(self):
        from sklearn.metrics import adjusted_rand_score
        from sklearn.cluster import KMeans
        
        selectedGroup = self.currentGroup.get()
        memberList = [member['name'] for member in self.groups[selectedGroup]]
        
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
        
        # Get all '_vocals.mp3' files
        vocalsOnlySongs = [
            f for f in os.listdir(audioDir)
            if f.endswith("_vocals.wav")
        ]
        
        if not jsonFiles or not vocalsOnlySongs:
            messagebox.showwarning("No Data", "No vocal or label files found.")
            return
        
        combineMemberVocals(jsonFiles, vocalsOnlySongs, currentGroup)
        messagebox.showinfo("Done", "Combined and saved member vocals")
    
    def chooseSongForHarmonyExtraction(self):
        """
        Opens a scrollable song list and extracts harmonies + segments
        for training when a song is clicked.
        """
        selectedGroup = self.currentGroup.get()

        def onSongSelected(songName):
            extractAndSaveHarmoniesFromSong(selectedGroup, songName)

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
            
if __name__ == "__main__":
    groups = {
            "IVE": [
                {'name': 'Gaeul', 'color': '#0000ff', 'priorTime': 18.3},
                {'name': 'Yujin', 'color': '#ff00ff', 'priorTime': 32.2},
                {'name': 'Rei', 'color': '#65bd2b', 'priorTime': 21.4},
                {'name': 'Wonyoung', 'color': '#ff0000', 'priorTime': 25.3},
                {'name': 'Liz', 'color': '#00c3f5', 'priorTime': 32.6},
                {'name': 'Leeseo', 'color': '#aa9f00', 'priorTime': 24.7}
            ],
            "ITZY": [
                {'name': 'Yeji', 'color': '#ffff00'}, 
                {'name': 'Lia', 'color': '#eb7d46'}, 
                {'name': 'Ryujin', 'color': '#7d46eb'},
                {'name': 'Chaeryeong', 'color': '#3232ff'}, 
                {'name': 'Yuna', 'color': '#46eb7d'}
            ],
            "BTS": [
                {"name": "Jin", "color": "#0000ff", "priorTime": 23.8},       
                {"name": "Suga", "color": "#46eb7d", "priorTime": 28.1},      
                {"name": "J-Hope", "color": "#eb46b4", "priorTime": 26.0}, 
                {"name": "RM", "color": "#eb7d46", "priorTime": 34.9},   
                {"name": "Jimin", "color": "#46b4eb", "priorTime": 38.2},  
                {"name": "V", "color": "#7d46eb", "priorTime": 31.6},
                {"name": "Jungkook", "color": "#ff0000", "priorTime": 62.4}
            ],
            "Fifty Fifty": [
                {'name': 'Keena', 'color': "#00852c"}, 
                {'name': 'Chanelle', 'color': '#8f69db'}, 
                {'name': 'Yewon', 'color': "#e98f11"},
                {'name': 'Hana', 'color': "#32ffee"}, 
                {'name': 'Athena', 'color': "#131dd7"}
            ]
    }   
    
    root = tk.Tk()
    
    app = VoiceTrainerGUI(root, groups)
    root.mainloop()