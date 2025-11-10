import numpy as np
import torch
from collections import Counter
import librosa

class VoicePredictor:
    def __init__(self, modelPath, memberList, groupName, songName='', contextSize=5, threshold=0.6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")

        # Load the trained PyTorch model
        self.model = None # WavLMClassifier(len(memberList), contextSize=contextSize).to(self.device)
        self.model.load_state_dict(torch.load(modelPath, map_location=self.device))
        self.model.eval()

        self.memberList = memberList
        self.groupName = groupName
        self.songName = songName
        self.contextSize = contextSize
        self.threshold = threshold
        self.silencePredictions = None
        
    def createContextMenu(self, chunks):
        contextHalf = self.contextSize // 2
        contextInputs = []

        for i in range(contextHalf, len(chunks) - contextHalf):
            window = chunks[i - contextHalf:i + contextHalf + 1]
            combined = np.concatenate(window, axis=0)  # (contextSize * T, F)
            contextInputs.append(combined)

        contextInputs = np.array(contextInputs)  # shape: (num_windows, contextSize * samplesPerChunk)
        contextInputs = np.expand_dims(contextInputs, axis=1)
        return np.array(contextInputs)
    
    def postprocessPrediction(self, prediction):
        confidentMembers = [
            self.memberList[i]
            for i, prob in enumerate(prediction)
            if prob >= self.threshold
        ]
        
        return confidentMembers if confidentMembers else ["None"]

    def majorityVote(self, labelList):
        smoothed = []
        pad = self.contextSize // 2
        padded = ['None'] * pad + labelList + ['None'] * pad

        for i in range(len(labelList)):
            window = padded[i:i+self.contextSize]
            flatLabels = [member for label in window for member in label if member != "None"]
            if not flatLabels:
                smoothed.append(["None"])
                continue
                
            # Count occurrences of each member
            counts = Counter(flatLabels)
            
            mostCommonMembers = []
            # Add all members that appear in at least 50% of the window
            for member, count in counts.items():
                if count >= (self.contextSize // 2 + 1):  # majority
                    mostCommonMembers.append(member)

            smoothed.append(mostCommonMembers if mostCommonMembers else ["None"])

        return smoothed

    def smoothPredictions(self, rawPredictions):
        smoothed = []
        padded = np.pad(rawPredictions, ((self.contextSize//2, self.contextSize//2), (0, 0)), mode='edge')

        for i in range(len(rawPredictions)):
            window = padded[i:i+self.contextSize]
            avg = np.mean(window, axis=0)  # Average across chunks
            smoothed.append(avg)

        return np.array(smoothed)
    
    def predictFromChunks(self, wavPath, chunkDurationMs=40, sampleRate=16000):
        print(f"Loading WAV file from {wavPath}...")

        # Load the raw audio waveform
        audio, _ = librosa.load(wavPath, sr=sampleRate, mono=True)
        print(f"Chunks loaded from audio...", len(audio))

        # Calculate number of samples per chunk
        samplesPerChunk = int(sampleRate * (chunkDurationMs / 1000.0))

        # Split into chunks
        chunks = [
            audio[i:i + samplesPerChunk]
            for i in range(0, len(audio), samplesPerChunk)
            if len(audio[i:i + samplesPerChunk]) == samplesPerChunk
        ]
        chunks = np.array(chunks)  # Shape: (num_chunks, samplesPerChunk)
        # create context windows
        contextInputs = self.createContextMenu(chunks)
        print(f"Total prediction windows: {contextInputs.shape[0]}")
        
        # Convert to PyTorch tensors
        inputs = torch.tensor(contextInputs, dtype=torch.float32).to(self.device)

        allPreds = []
        with torch.no_grad():
            for i in range(0, len(inputs), 64):  # batch inference
                batch = inputs[i:i+64]
                preds = self.model(batch)
                allPreds.append(preds.cpu().numpy())
        rawSingers = np.concatenate(allPreds, axis=0)  # shape: (num_windows, num_members)
        
        # Smoothing predictions across context windows
        predictions = self.smoothPredictions(rawSingers)

        # Final labels for each 40ms chunk
        offset = self.contextSize // 2
        numChunks = len(chunks)
        finalLabels = [["None"]] * numChunks  # Default to "None"

        for i, pred in enumerate(predictions):
            chunkIndex = i + offset
            if chunkIndex < numChunks:
                voices = self.postprocessPrediction(pred)
                finalLabels[chunkIndex] = voices

        return finalLabels, predictions
    
    def cleanShortSingers(self, predictions, minStreak=8):
        """
        Remove singers who never appear for at least `minStreak` consecutive chunks.

        Parameters:
            predictions (list of list of str): Each entry is a list of singer names for a chunk,
                                            e.g., [["Leeseo"], ["Leeseo", "Wonyoung"], ...]

            minStreak (int): Minimum number of consecutive detections required to keep a singer

        Returns:
            list of list of str: Cleaned predictions with short streak singers removed
        """
        from collections import defaultdict
         # Track current streaks and all short ranges to delete
        currentStreaks = defaultdict(lambda: {"start": None, "length": 0})
        rangesToRemove = defaultdict(list)

        for idx, chunk in enumerate(predictions):
            currentSingers = set(chunk)

            # End any streaks for singers no longer in current chunk
            for singer in list(currentStreaks.keys()):
                if singer not in currentSingers and currentStreaks[singer]["length"] > 0:
                    start = currentStreaks[singer]["start"]
                    length = currentStreaks[singer]["length"]
                    if start is not None and length <= minStreak:
                        rangesToRemove[singer].append((start, idx - 1))
                    currentStreaks[singer]["start"] = None
                    currentStreaks[singer]["length"] = 0

            # Continue streaks for singers in current chunk
            for singer in currentSingers:
                if currentStreaks[singer]["length"] == 0:
                    currentStreaks[singer]["start"] = idx
                currentStreaks[singer]["length"] += 1

        # Handle any lingering streaks at the end
        for singer, data in currentStreaks.items():
            if data["length"] > 0 and data["length"] <= minStreak and data["start"] is not None:
                endIdx = data["start"] + data["length"] - 1
                rangesToRemove[singer].append((data["start"], endIdx))

        # Deep copy predictions
        cleaned = [chunk.copy() for chunk in predictions]

        # Apply deletions based on ranges
        for singer, ranges in rangesToRemove.items():
            for start, end in ranges:
                for i in range(start, end + 1):
                    if singer in cleaned[i]:
                        cleaned[i] = [s for s in cleaned[i] if s != singer]

        # Replace empty chunks with ["None"]
        for i in range(len(cleaned)):
            if not cleaned[i]:
                cleaned[i] = ["None"]

        return cleaned
    
    def evaluateSilencePredictions(self, trueSilenceMask):
        """
        Compare predicted silence mask (self.silencePredictions) with ground truth silence mask.
        trueSilenceMask: array of 0 (singing) and 1 (silence)
        """
        if self.silencePredictions is None:
            print("❌ No silence predictions found. Run predictFromChunks() first.")
            return

        if len(self.silencePredictions) != len(trueSilenceMask):
            print(f"❌ Mismatch in lengths: predicted {len(self.silencePredictions)}, ground truth {len(trueSilenceMask)}")
            return

        pred = self.silencePredictions
        true = np.array(trueSilenceMask)

        tp = np.sum((pred == 1) & (true == 1))  # True silence correctly detected
        tn = np.sum((pred == 0) & (true == 0))  # True singing correctly detected
        fp = np.sum((pred == 1) & (true == 0))  # Mistakenly called singing silence
        fn = np.sum((pred == 0) & (true == 1))  # Missed silence (called silence singing)

        total = len(true)

        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"🔎 Silence Detection Evaluation:")
        print(f"Total Chunks: {total}")
        print(f"True Positives (silence correctly detected): {tp}")
        print(f"True Negatives (singing correctly detected): {tn}")
        print(f"False Positives (wrongly detected silence): {fp}")
        print(f"False Negatives (missed silence): {fn}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
    
    def setVocalActivityMask(self, silence, minGap=5):
        """
        Convert raw multi-label predictions into binary vocal activity mask.
        
        Args:
            predictions (np.ndarray): shape (numChunks, numMembers) – raw output probs
            threshold (float): Probability threshold to consider a member "active"
            minGap (int): Minimum number of silence chunks to preserve between vocals

        Returns:
            np.ndarray: Binary mask (0 = silence, 1 = member singing)
        """
        # print("Pure silence:", silence[250:370])
        # Fill short 0 gaps between 1's
        i = 0
        resultMask = silence.copy()
        while i < len(resultMask):
            if resultMask[i] == 0:
                gapStart = i
                while i < len(resultMask) and resultMask[i] == 0:
                    i += 1
                gapEnd = i
                
                # only fill gap if it's short and surrounded by 's
                if gapStart > 0 and gapEnd < len(resultMask) and (gapEnd - gapStart) < minGap:
                    resultMask[gapStart:gapEnd] = 1
            else:
                i += 1

        return resultMask 
    
    def generateSilenceMaskFromLabels(self, labelChunks, totalChunks):
        """
        Generate a silence mask based on member singing labels.
        
        Args:
            labelChunks (list): List of [memberName, startChunk, endChunk].
            totalChunks (int): Total number of chunks in the song.

        Returns:
            np.ndarray: Array of shape (totalChunks,), where
                        1 = silence (no member singing),
                        0 = singing (at least one member labeled)
        """
        mask = np.ones(totalChunks, dtype=int)  # Start assuming everything is silence (1)

        for label in labelChunks:
            member, start, end = label
            # Make sure start and end are within bounds
            start = max(0, start)
            end = min(totalChunks - 1, end)
            
            mask[start:end+1] = 0  # Mark as singing

        return mask


    def savePredictions(self, predictions, savePath, rawPredictions=None):
        with open(savePath, "w", encoding="utf-8") as f:
            for index, voices in enumerate(predictions):
                line = f"{index}: {', '.join(voices)}"
                
                if rawPredictions is not None and index < len(rawPredictions):
                    raw = rawPredictions[index]
                    probs = ', '.join([
                        f"{self.memberList[i]}: {raw[i]:.2f}" for i in range(len(self.memberList))
                    ])
                    line += f"  |  Probabilities: [{probs}]"
                
                f.write(line + "\n")
        
        print(f"✅ Predictions saved to {savePath}")