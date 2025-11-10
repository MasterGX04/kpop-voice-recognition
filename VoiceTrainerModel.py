import numpy as np
from threading import Thread
from tqdm import tqdm
from queue import Queue
from collections import Counter 
from audio_processing import augmentChunkLibrosa, estimatePitchRanges
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, Add, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ProgbarLogger, EarlyStopping
from tensorflow.keras.models import Model, load_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
from perceptron_hmm import PerceptronHMMTrainer

class VoiceModelTrainer:
    def __init__(self, features, labels, rawChunks, labelRanges, groupName, contextSize=3, stage="mixed"):
        self.features = features
        self.labels = labels
        self.rawChunks = rawChunks
        self.labelRanges = labelRanges
        self.contextSize = contextSize
        self.groupName = groupName
        self.xTrain = None
        self.xTrainPitch = None
        self.xVal = None
        self.yTrain = None
        self.yVal = None
        self.model = None
        self.sampleWeights = None
        self.stage = stage
        self.chunkDuration = 40
        
    def buildContextWindows(self):
        self.nMels = self.features[0].shape[1]
        print(f"Using feature dimension: {self.nMels}")
        
        def createSoftLabel(labelWindow):
            softLabel = np.mean(labelWindow, axis=0)  # Average one-hot vectors
            return softLabel
                
        print("Building context menu")
        halfWindow = self.contextSize // 2
        totalChunks = len(self.features)
        
        # Create repeat/adlib masks using self.labelRanges
        repeatMask = np.zeros(totalChunks, dtype=bool)
        adLibMask = np.zeros(totalChunks, dtype=bool)
        
        for entry in self.labelRanges:
            if len(entry) < 3: continue
            
            _, start, end, isRepeat, isAdLib = entry
            start = max(0, start)
            end = min(totalChunks - 1, end)
            if isRepeat:
                repeatMask[start:end + 1] = True
            if isAdLib:
                adLibMask[start:end + 1] = True

        contextFeatures, contextLabels, contextSilenceLabels, sampleWeights, rawChunkData = [], [], [], [], []
        
        memberCounts = Counter()

        for i in tqdm(range(halfWindow, totalChunks - halfWindow), desc="Processing windows"):
            window = self.features[i - halfWindow:i + halfWindow + 1]
            labelWindow = self.labels[i - halfWindow:i + halfWindow + 1]

            chunkSingerCounts = [np.count_nonzero(chunk) for chunk in labelWindow]
            isRepeatChunk = np.any(repeatMask[i - halfWindow:i + halfWindow + 1])
            isAdLibChunk = np.any(adLibMask[i - halfWindow:i + halfWindow + 1])
            numSilentChunks = sum(count == 0 for count in chunkSingerCounts)
            isSilence = (numSilentChunks / len(labelWindow)) >= 0.8

            if self.stage == "solo" and isSilence:
                continue  # skip silence in solo training

            try:
                combined = np.concatenate(window, axis=0)
                softLabel = createSoftLabel(labelWindow)

                if self.stage == "solo" and np.sum(softLabel >= 0.5) != 1:
                    continue  # skip multi-singer or ambiguous solo chunk

                # Tally dominant member
                dominantIdx = np.argmax(softLabel)
                if softLabel[dominantIdx] >= 0.5:
                    memberCounts[dominantIdx] += 1

                rawChunkData.append((combined, softLabel, isSilence, dominantIdx, isRepeatChunk, isAdLibChunk))

            except Exception as e:
                print(f"Skipped index {i} due to error: {e}")
                continue   
                
        print("\n🎤 Member Chunk Distribution:")
        maxChunks = max(memberCounts.values())
        memberWeights = {idx: maxChunks / count for idx, count in memberCounts.items()}
        for idx, count in sorted(memberCounts.items()):
            print(f"  Member {idx}: {count} chunks") 
            
        # Calculate chunk ratios for naive bayes
        totalChunks = sum(memberCounts.values())
        memberChunkRatios = {
            idx: count / totalChunks for idx, count in memberCounts.items()
        }
        maxRatio = max(memberChunkRatios.values()) # For Normalization
        
        print("Estimating pitch ranges for raw chunk data...")
        # Add multithreading here
        if self.rawChunks is None:
            raise ValueError("rawChunks is required to estimate pitch ranges.")
        savePath = f"./{self.groupName}/{self.groupName}_pitch_ranges.npy"
        pitchRanges = estimatePitchRanges(audioChunks=self.rawChunks, groupName=self.groupName, groupSize=self.contextSize, savePath=savePath)
        
        # Pitch vectors to use as auxillary
        pitchVectors = []
        from collections import defaultdict
        # Tally counts per member per range
        memberRegisterCounts = defaultdict(lambda: {"low": 1, "mid": 1, "high": 1})
        for i, (_, _, _, dominantIdx, *_) in enumerate(rawChunkData):
            pitch = pitchRanges[i]
            if pitch in {"low", "mid", "high"}:
                memberRegisterCounts[dominantIdx][pitch] += 1
        
        # Calculate pitch distribution ratios
        memberPitchRatios = {
            idx: {k: v / sum(counts.values()) for k, v in counts.items()}
            for idx, counts in memberRegisterCounts.items()
        }
        for idx, ratios in memberPitchRatios.items():
            print(f"Member {idx} pitch ratios: {ratios}")
        
        # Compute global pitch distribution once
        globalPitchCounts = {"low": 0, "mid": 0, "high": 0}
        for counts in memberRegisterCounts.values():
            for pitchType in globalPitchCounts:
                globalPitchCounts[pitchType] += counts[pitchType]
                    
        totalGlobal = sum(globalPitchCounts.values())
        globalPitchRatios = {k: globalPitchCounts[k] / totalGlobal for k in globalPitchCounts}
        
        print(f"Global Pitch Ratios: {globalPitchRatios}")
        # Finalize features and weights
        for i, (combined, softLabel, isSilence, dominantIdx, isRepeatChunk, isAdLibChunk) in enumerate(rawChunkData):
            contextFeatures.append(combined)
            contextLabels.append(softLabel)
            contextSilenceLabels.append(np.array([1.0]) if isSilence else np.array([0.0]))
            
            pitchLabel = pitchRanges[i]
            pitchOneHot = {
                "low": [1, 0, 0],
                "mid": [0, 1, 0],
                "high": [0, 0, 1],
            }.get(pitchLabel, [0, 0, 0])
            pitchVectors.append(pitchOneHot)
            
            if self.stage == "solo" and memberCounts[dominantIdx] < 8000:
                timeSteps = combined.shape[0] // self.nMels
                if timeSteps > 0:
                    reshapeShape = (self.nMels * timeSteps,)
                    augmented = augmentChunkLibrosa(combined, sr=22050, n_mels=self.nMels, reshapeShape=reshapeShape)
                    if augmented.shape == combined.shape:
                        contextFeatures.append(augmented)
                        contextLabels.append(softLabel)
                        contextSilenceLabels.append(np.array([1.0]) if isSilence else np.array([0.0]))
                        pitchVectors.append(pitchOneHot) 
                        
            baseWeight = memberWeights.get(dominantIdx, 1.0)
            
            # List weight based on how rare pitch is for singer
            if pitchLabel in ["low", "high"]:
                memberRatio = memberPitchRatios.get(dominantIdx, {}).get(pitchLabel, 1e-6)
                globalRatio = globalPitchRatios.get(pitchLabel, 1e-6)
                
                if memberRatio < globalRatio:
                    # underrepresented -> boost
                    boost = globalRatio / memberRatio
                    baseWeight *= min(2.5, boost)
                else:
                    # Overrepresented -> soften
                    penalty = globalRatio / memberRatio
                    baseWeight *= max(0.7, 1.0 / penalty)
            
            # More weight for underrepresented members and less for overrepresented members
            classPenalty = memberChunkRatios.get(dominantIdx, 1e-6)
            classBoost = maxRatio / classPenalty
            baseWeight *= min(2.0, classBoost) # Cap at 2.0 to avoid overboosting
            
            # Apply existing penalties
            if isRepeatChunk:
                baseWeight *= 0.8
            elif isAdLibChunk:
                baseWeight *= 0.6

            sampleWeights.append(baseWeight)
                
        self.x = [np.array(contextFeatures), np.array(pitchVectors)]
        self.y = [np.array(contextLabels), np.array(contextSilenceLabels)]
        self.sampleWeights = np.array(sampleWeights)

    def splitDataset(self, testSize=0.2, randomState=42):
        mfccs, pitchVectors = self.x
        ySingers, ySilence = self.y

        # Split both label parts alongside x
        # Use same random split for all
        (
            xTrainMfcc, xValMfcc,
            xTrainPitch, xValPitch,
            yTrainSingers, yValSingers,
            yTrainSilence, yValSilence
        ) = train_test_split(
            mfccs, pitchVectors, ySingers, ySilence,
            test_size=testSize,
            random_state=randomState
        )

        self.xTrain = [xTrainMfcc, xTrainPitch]
        self.xVal = [xValMfcc, xValPitch]
        self.yTrain = [yTrainSingers, yTrainSilence]
        self.yVal = [yValSingers, yValSilence]

    def buildModel(self, inputShapes, outputSize):
        print("Model input shape:", inputShapes) 
        inputMfccShape, inputPitchShape = inputShapes 
        # Main input (MFCC Features)
        mfccInput = Input(shape=inputMfccShape, name="mfcc_input")      # e.g. (7, 153)
        pitchInput = Input(shape=inputPitchShape, name="pitch_input")
    
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(mfccInput)
        x = Dropout(0.2)(x)
        x = transformerEncoder(x, numHeads=4, ffDim=128, dropout=0.3)
        x = GlobalAveragePooling1D()(x)  # Pool across time
        x = Dropout(0.4)(x)
        
        # 🔗 Concatenate pitch vector
        x = tf.keras.layers.Concatenate()([x, pitchInput])
        
        # Dense head
        reg = regularizers.l2(1e-4) 
        embedding = Dense(32, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name='embedding')(x)
        multiLabelOutput = Dense(outputSize, activation='sigmoid', kernel_regularizer=reg, bias_regularizer=reg, name='singer_output')(embedding)
        silenceOutput = Dense(1, activation='sigmoid', kernel_regularizer=reg, bias_regularizer=reg, name='silence_output')(embedding)

        baseModel = Model([mfccInput, pitchInput], [multiLabelOutput, silenceOutput])
        model = CustomVoiceModel(baseModel, alpha=0.1)
        
        # losses = {
        #     "singer_output": model.lossFn,
        #     "silence_output": tf.keras.losses.BinaryCrossentropy()
        # }
        
        # Add label smoothing to binary crossentropy
        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())
        self.model = model
    
    def trainModel(self, batchSize=32, epochs=30):
        if self.model is None:
            raise ValueError("Model not built yet.")
        earlyStop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        # ADd this later: [self.xTrain, self.xTrainPitch]
        self.model.fit(
            self.xTrain,
            self.yTrain,
            validation_data=(self.xVal, self.yVal),
            epochs=epochs,
            batch_size=batchSize,
            callbacks=[earlyStop]
        )
        
    def runPipeline(self, skipBuildModel=False):
        self.buildContextWindows()
        numSilence = np.sum([1 for s in self.y[1] if s[0] == 1.0])
        numSinging = len(self.y[1]) - numSilence
        print(f"Silence: {numSilence}, Singing: {numSinging}")
        
        self.splitDataset()
            
        # Step 1: Get input shape (manually extract one valid chunk to get shape)
        half = self.contextSize // 2
        sampleChunks = self.features[half - half:half + half + 1]
        sample = np.concatenate(sampleChunks, axis=0)

        # if os.path.exists(self.encoderDir):
        #     print(f"✅ Found existing encoder at {self.encoderDir}. Skipping pretraining.")
        # else:
        #     print(f"🧠 No encoder found at {self.encoderDir}. Initiating self-supervised pretraining.")
        #     pretrainer = VoiceSelfSupervisedTrainer(
        #         features=self.features,
        #         labels=self.labels,
        #         contextSize=self.contextSize,
        #         inputShape=inputShape
        #     )
        #     pretrainer.buildSiameseModel()
        #     pretrainer.train()
        #     pretrainer.encoder.trainable = False
        #     pretrainer.saveEncoder(self.encoderDir)
            
        if not skipBuildModel:
            self.buildModel(
                inputShapes=(self.xTrain[0].shape[1:], self.xTrain[1].shape[1:]),
                outputSize=self.yTrain[0].shape[1]
            )
        self.trainModel()
        
        # Train perceptron-HMM
        flatX = [np.concatenate([mfcc.flatten(), pitch]) for mfcc, pitch in zip(self.x[0], self.x[1])]
        trainer = PerceptronHMMTrainer(flatX, self.y[0], list(range(self.y[0].shape[1])))
        trainer.buildAndTrainMLP()
        trainer.trainHMM()
        self.perceptronHMM = trainer

def transformerEncoder(inputs, numHeads=4, ffDim=128, dropout=0.1):
    # Self-attention
    attentionOutput = MultiHeadAttention(num_heads=numHeads, key_dim=inputs.shape[-1])(inputs, inputs)
    attentionOutput = Dropout(dropout)(attentionOutput)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([inputs, attentionOutput]))
    
    # Feedforward
    ffn = Dense(ffDim, activation="relu")(out1)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn = Dropout(dropout)(ffn)
    return LayerNormalization(epsilon=1e-6)(Add()([out1, ffn]))

class CustomVoiceModel(Model):
    def __init__(self, baseModel, alpha=0.1):
        super().__init__()
        self.baseModel = baseModel
        self.alpha = alpha  # weight for entropy penalty
        self.lossFn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
        
        self.singerAccuracy = tf.keras.metrics.BinaryAccuracy(name="singer_accuracy")
        self.silenceAccuracy = tf.keras.metrics.BinaryAccuracy(name="silence_accuracy")

    def train_step(self, data):
        x, y = data
        if isinstance(y, tuple) or isinstance(y, list):
            ySingers, ySilence = y
        else:
            raise ValueError("Expected tuple (singer_labels, silence_labels)")

        with tf.GradientTape() as tape:
            yPredSingers, yPredSilence = self.baseModel(x, training=True)
            
            lossSinger = self.lossFn(ySingers, yPredSingers)
            lossSilence = BinaryCrossentropy()(ySilence, yPredSilence)

            # Entropy regularization for overlapping labels
            probs = tf.clip_by_value(yPredSingers, 1e-7, 1.0)
            overlapMask = tf.cast(tf.reduce_sum(ySingers, axis=-1) > 1.0, tf.float32)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs), axis=1)
            entropyLoss = tf.reduce_mean(entropy * overlapMask)

            totalLoss = lossSinger + 0.3 * lossSilence - self.alpha * entropyLoss

        grads = tape.gradient(totalLoss, self.baseModel.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.baseModel.trainable_variables))
        
        self.singerAccuracy.update_state(ySingers, yPredSingers)
        self.silenceAccuracy.update_state(ySilence, yPredSilence)
        
        return {
            "loss": totalLoss,
            "singer_accuracy": self.singerAccuracy.result(),
            "silence_accuracy": self.silenceAccuracy.result(),
        }
    
    def call(self, inputs):
        return self.baseModel(inputs, training=False)
    
    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs) 