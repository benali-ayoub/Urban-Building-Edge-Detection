class ImageMetrics:
    def __init__(self):
        self.metrics = {}

    def calculate_all_metrics(self, segmented_image, ground_truth):
        """Calculate metrics with focus on edge quality"""
        try:
            # Ensure binary images
            seg_binary = (segmented_image > 0).astype(np.uint8)
            gt_binary = (ground_truth > 0).astype(np.uint8)

            # Calculate pixel-wise metrics
            seg_flat = seg_binary.flatten()
            gt_flat = gt_binary.flatten()
            
            # Calculate edge alignment using distance transform
            dist_seg = cv2.distanceTransform(1 - seg_binary, cv2.DIST_L2, 3)
            dist_gt = cv2.distanceTransform(1 - gt_binary, cv2.DIST_L2, 3)
            
            max_dist = np.maximum(dist_seg.max(), dist_gt.max())
            dist_seg_norm = dist_seg / (max_dist + 1e-10)
            dist_gt_norm = dist_gt / (max_dist + 1e-10)
            edge_alignment = 1.0 - np.mean(np.abs(dist_seg_norm - dist_gt_norm))

            # Calculate edge thickness similarity
            thickness_ratio = np.sum(seg_binary) / (np.sum(gt_binary) + 1e-10)
            thickness_score = np.exp(-np.abs(thickness_ratio - 1.0))

            self.metrics = {
                'accuracy': accuracy_score(gt_flat, seg_flat),
                'precision': precision_score(gt_flat, seg_flat, zero_division=1),
                'recall': recall_score(gt_flat, seg_flat, zero_division=1),
                'f1': f1_score(gt_flat, seg_flat, zero_division=1),
                'ssim': ssim(ground_truth, segmented_image),
                'edge_alignment': edge_alignment,
                'thickness_score': thickness_score
            }

            return self.metrics

        except Exception as e:
            print(f"Error in metric calculation: {str(e)}")
            return {metric: 0.0 for metric in 
                   ['accuracy', 'precision', 'recall', 'f1', 'ssim', 
                    'edge_alignment', 'thickness_score']}

    def get_weighted_score(self):
        """Calculate weighted score emphasizing edge quality"""
        try:
            weights = {
                'accuracy': 0.15,
                'precision': 0.15,
                'recall': 0.15,
                'f1': 0.15,
                'ssim': 0.15,
                'edge_alignment': 0.15,
                'thickness_score': 0.10
            }
            
            base_score = sum(self.metrics[metric] * weight 
                           for metric, weight in weights.items())
            
            # Bonus for good edge alignment and thickness
            if self.metrics['edge_alignment'] > 0.8 and self.metrics['thickness_score'] > 0.8:
                base_score *= 1.2
            
            # Penalty for poor edge quality
            if self.metrics['edge_alignment'] < 0.5 or self.metrics['thickness_score'] < 0.5:
                base_score *= 0.8
            
            return max(0.0, min(1.0, base_score))

        except Exception as e:
            print(f"Error in reward calculation: {str(e)}")
            return 0.0