import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GuitarChordDistanceLoss(nn.Module):
    """
    Loss function for guitar chord recognition with sin/cos encoding.
    
    Handles:
    - 12 root notes in chromatic circle (A, A#, B, C, C#, D, D#, E, F, F#, G, G#)
    - Major/minor quality
    - Noise class (treated as maximally distant from all chords)
    
    Args:
        label_to_index: Dict mapping chord names to indices
        alpha: Weight for distance penalty (0-1). 0 = pure CE, 1 = pure distance
        root_weight: How much to weight root (0.7 = 70% root, 30% quality)
        temperature: Softens distance penalty (higher = more forgiving)
        noise_distance: Fixed distance from noise to any chord (default: 2.0 = max)
        class_weights: Optional class weights for imbalanced data
    """
    def __init__(self, label_to_index, alpha=0.3, root_weight=0.7, 
                 temperature=2.0, noise_distance=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.root_weight = root_weight
        self.temperature = temperature
        self.noise_distance = noise_distance
        
        # Parse all chord labels
        self.chord_info = self._parse_guitar_chords(label_to_index)
        n_classes = len(label_to_index)
        
        # Create distance matrix with sin/cos encoding
        self.register_buffer('distance_matrix', 
                           self._create_distance_matrix(n_classes))
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    
    def _parse_guitar_chords(self, label_to_index):
        """
        Parse your exact chord format: A, A#, A#m, Am, ..., Noise
        """
        # Chromatic scale starting from A
        note_to_position = {
            'A': 0, 'A#': 1, 'B': 2, 'C': 3, 'C#': 4, 'D': 5,
            'D#': 6, 'E': 7, 'F': 8, 'F#': 9, 'G': 10, 'G#': 11
        }
        
        chord_info = {}
        
        for label, idx in label_to_index.items():
            if label == 'Noise':
                chord_info[idx] = {
                    'type': 'noise',
                    'root_position': None,
                    'quality': None,
                    'label': label
                }
            else:
                # Parse: root note + optional 'm' for minor
                is_minor = label.endswith('m')
                root = label[:-1] if is_minor else label
                
                chord_info[idx] = {
                    'type': 'chord',
                    'root_position': note_to_position[root],
                    'quality': 'min' if is_minor else 'maj',
                    'label': label
                }
        
        return chord_info
    
    def _create_distance_matrix(self, n_classes):
        """
        Create distance matrix using:
        1. Sin/cos encoding for root notes (circular)
        2. Binary distance for quality (0 or 1)
        3. Fixed maximum distance for Noise class
        """
        dist_matrix = torch.zeros(n_classes, n_classes)
        
        for i in range(n_classes):
            for j in range(n_classes):
                info_i = self.chord_info[i]
                info_j = self.chord_info[j]
                
                # Handle Noise class
                if info_i['type'] == 'noise' or info_j['type'] == 'noise':
                    if info_i['type'] == 'noise' and info_j['type'] == 'noise':
                        # Noise to Noise
                        dist_matrix[i, j] = 0.0
                    else:
                        # Noise to any chord: maximum distance
                        dist_matrix[i, j] = self.noise_distance
                    continue
                
                # Both are chords: compute musical distance
                pos_i = info_i['root_position']
                pos_j = info_j['root_position']
                
                # 1. Root distance using circular (angular) distance
                # Calculate shortest path around the circle
                semitone_diff = abs(pos_i - pos_j)
                circular_distance = min(semitone_diff, 12 - semitone_diff)
                
                # Normalize to [0, 1]: max distance is 6 semitones (tritone)
                root_dist = circular_distance / 6.0
                
                # 2. Quality distance
                quality_dist = 0.0 if info_i['quality'] == info_j['quality'] else 1.0
                
                # 3. Weighted combination
                total_dist = (self.root_weight * root_dist + 
                            (1 - self.root_weight) * quality_dist)
                
                dist_matrix[i, j] = total_dist
        
        return dist_matrix
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, T, C) or (B*T, C) - raw model outputs
            targets: (B, T) or (B*T,) - ground truth chord indices
        Returns:
            loss: scalar tensor
        """
        original_shape = logits.shape
        if len(original_shape) == 3:
            B, T, C = original_shape
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)
        
        # 1. Standard cross-entropy loss
        ce_loss = self.ce_loss(logits, targets)
        
        # 2. Distance-weighted penalty
        probs = F.softmax(logits, dim=-1)  # (N, C)
        
        # Get distance from true class to all predicted classes
        target_distances = self.distance_matrix[targets]  # (N, C)
        
        # Apply temperature to soften the penalty curve
        distance_penalty = (target_distances / self.temperature) ** 2
        
        # Weight by prediction confidence (penalize confident wrong predictions more)
        weighted_penalty = (probs * distance_penalty).sum(dim=-1)
        
        # 3. Combine losses
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * weighted_penalty
        
        return total_loss.mean()
    
    def get_distance_visualization(self):
        """
        Returns distance matrix and labels for visualization.
        Useful for debugging and understanding the distance structure.
        """
        n = len(self.chord_info)
        labels = [self.chord_info[i]['label'] for i in range(n)]
        
        return {
            'labels': labels,
            'distance_matrix': self.distance_matrix.cpu().numpy(),
            'chord_info': self.chord_info
        }
    
    def print_example_distances(self):
        """Print some example distances to verify they make sense"""
        examples = [
            ('C', 'C'),      # Same chord
            ('C', 'Cm'),     # Same root, different quality
            ('C', 'D'),      # Adjacent major chords
            ('C', 'G'),      # Perfect 5th (7 semitones)
            ('C', 'F#'),     # Tritone (6 semitones, opposite on circle)
            ('C', 'Am'),     # Relative minor (3 semitones, different quality)
            ('C', 'Noise'),  # Chord to noise
            ('Noise', 'Noise'),  # Noise to itself
        ]
        
        print("\nExample chord distances:")
        print("-" * 50)
        
        for label1, label2 in examples:
            idx1 = None
            idx2 = None
            
            # Find indices
            for idx, info in self.chord_info.items():
                if info['label'] == label1:
                    idx1 = idx
                if info['label'] == label2:
                    idx2 = idx
            
            if idx1 is not None and idx2 is not None:
                dist = self.distance_matrix[idx1, idx2].item()
                print(f"{label1:6s} → {label2:6s}: {dist:.3f}")
        
        print("-" * 50)


# Example usage and integration:
"""
# In your train.py, replace the criterion line:

from src.loss import GuitarChordDistanceLoss

# Inside run() function:
criterion = GuitarChordDistanceLoss(
    label_to_index=train_ds.label_to_index,
    alpha=0.3,              # 30% distance penalty, 70% cross-entropy
    root_weight=0.7,        # 70% root importance, 30% quality importance
    temperature=2.0,        # Smoothing factor
    noise_distance=2.0,     # Noise is maximally far from all chords
    class_weights=class_weights
).to(DEVICE)

# Optional: Print example distances to verify
criterion.print_example_distances()

# Optional: Visualize the full distance matrix
import matplotlib.pyplot as plt
import numpy as np

viz = criterion.get_distance_visualization()
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(viz['distance_matrix'], cmap='viridis', aspect='auto')

# Set ticks
ax.set_xticks(range(len(viz['labels'])))
ax.set_yticks(range(len(viz['labels'])))
ax.set_xticklabels(viz['labels'], rotation=90, fontsize=8)
ax.set_yticklabels(viz['labels'], fontsize=8)

# Add colorbar
plt.colorbar(im, ax=ax, label='Distance')
plt.title('Guitar Chord Distance Matrix (Sin/Cos Encoding)', fontsize=12)
plt.tight_layout()
plt.savefig('chord_distance_matrix.png', dpi=150)
print("Distance matrix saved to chord_distance_matrix.png")

# Then use the criterion normally in your training loop:
# loss = criterion(logits, y)
"""


# Hyperparameter tuning guide:
"""
TUNING RECOMMENDATIONS:

1. alpha (distance penalty weight):
   - Start: 0.2-0.3
   - If model predicts too "safely" (always nearby chords): increase to 0.4-0.5
   - If accuracy drops: decrease to 0.1-0.2
   - Range: [0.1, 0.5]

2. root_weight (root vs quality importance):
   - Start: 0.7 (root is more important)
   - If C↔Cm confusion is worse than C↔D confusion: decrease to 0.5-0.6
   - If C↔D confusion is worse than C↔Cm confusion: increase to 0.8-0.9
   - Range: [0.5, 0.9]

3. temperature (penalty softness):
   - Start: 2.0
   - Higher (3.0): more forgiving, smoother gradients
   - Lower (1.0): stricter penalty, sharper gradients
   - Range: [1.0, 3.0]

4. noise_distance:
   - Start: 2.0 (maximum distance)
   - This ensures Noise is treated as completely different from all chords
   - Usually don't need to change this

EXPECTED BEHAVIOR:
- Same chord (C→C): distance ≈ 0.0
- Same root, different quality (C→Cm): distance ≈ 0.3
- Adjacent roots, same quality (C→D): distance ≈ 0.1-0.2
- Opposite on circle (C→F#): distance ≈ 0.7-0.8
- Any chord to Noise: distance = 2.0
"""