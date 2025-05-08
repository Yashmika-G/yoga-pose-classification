import numpy as np
import cv2
import mediapipe as mp

class PoseAnalyzer:
    def __init__(self):
        # Reference angles and proportions for each pose
        self.pose_references = {
            'warrior2': {
                'description': 'Warrior II Pose (Virabhadrasana II)',
                'key_points': {
                    'arms_horizontal': {'threshold': 15, 'weight': 0.3},
                    'front_knee_angle': {'target': 90, 'threshold': 20, 'weight': 0.3},
                    'torso_upright': {'threshold': 10, 'weight': 0.2},
                    'back_leg_straight': {'target': 170, 'threshold': 20, 'weight': 0.2}
                },
                'guidance': [
                    "Stand with feet 3-4 feet apart, front foot pointing forward",
                    "Extend arms parallel to the floor, reaching in opposite directions",
                    "Bend your front knee to 90 degrees, keeping knee above ankle",
                    "Keep back leg straight and strong with foot at 90 degrees",
                    "Keep torso upright, don't lean forward or backward",
                    "Gaze over the front hand, shoulders relaxed"
                ]
            },
            'tree': {
                'description': 'Tree Pose (Vrksasana)',
                'key_points': {
                    'standing_leg_straight': {'target': 170, 'threshold': 15, 'weight': 0.3},
                    'hip_openness': {'threshold': 15, 'weight': 0.3},
                    'torso_upright': {'threshold': 10, 'weight': 0.2},
                    'foot_placement': {'threshold': 10, 'weight': 0.2}
                },
                'guidance': [
                    "Stand on one leg with the other foot placed against inner thigh (never on knee)",
                    "Keep hips level and facing forward",
                    "Keep your standing leg straight but not locked",
                    "Bring palms together at heart center or extend arms overhead",
                    "Fix your gaze on a non-moving point for balance",
                    "Keep your spine straight and shoulders relaxed"
                ]
            },
            'plank': {
                'description': 'Plank Pose',
                'key_points': {
                    'body_straight_line': {'threshold': 15, 'weight': 0.4},
                    'shoulder_elbow_alignment': {'threshold': 15, 'weight': 0.3},
                    'head_alignment': {'threshold': 15, 'weight': 0.15},
                    'hip_height': {'threshold': 15, 'weight': 0.15}
                },
                'guidance': [
                    "Start in a push-up position with arms straight",
                    "Form a straight line from head to heels",
                    "Keep shoulders directly above wrists",
                    "Engage your core and keep hips level (not sagging or raised too high)",
                    "Look slightly forward to keep neck in neutral position",
                    "Spread fingers wide and press into the floor"
                ]
            },
            'goddess': {
                'description': 'Goddess Pose (Utkata Konasana)',
                'key_points': {
                    'feet_position': {'threshold': 15, 'weight': 0.2},
                    'knee_angle': {'target': 90, 'threshold': 20, 'weight': 0.3},
                    'knee_alignment': {'threshold': 15, 'weight': 0.3},
                    'torso_upright': {'threshold': 10, 'weight': 0.2}
                },
                'guidance': [
                    "Stand with feet wide apart, toes pointing outward at 45 degrees",
                    "Bend knees directly over ankles to create 90-degree angle",
                    "Keep torso upright, tailbone tucked, core engaged",
                    "Bring arms to goal post position or extend to sides",
                    "Keep shoulders relaxed away from ears",
                    "Look straight ahead with chin parallel to floor"
                ]
            },
            'downdog': {
                'description': 'Downward Facing Dog (Adho Mukha Svanasana)',
                'key_points': {
                    'inverted_v_shape': {'threshold': 20, 'weight': 0.3},
                    'arms_straight': {'target': 170, 'threshold': 20, 'weight': 0.2},
                    'legs_straight': {'target': 160, 'threshold': 25, 'weight': 0.2},
                    'shoulder_alignment': {'threshold': 15, 'weight': 0.15},
                    'head_alignment': {'threshold': 15, 'weight': 0.15}
                },
                'guidance': [
                    "Start on hands and knees, then lift hips upward",
                    "Form an inverted V-shape with your body",
                    "Straighten legs as much as comfortable (slight bend is fine)",
                    "Press chest toward thighs, heels toward floor",
                    "Arms straight with shoulders away from ears",
                    "Head between arms, neck relaxed"
                ]
            }
        }
        
        # MediaPipe pose landmark indices
        self.mp_pose = mp.solutions.pose
        
    def analyze_pose(self, pose_name, landmarks):
        """
        Analyze the detected pose and provide feedback based on the landmark positions
        """
        if not landmarks or pose_name not in self.pose_references:
            return None, 0.0, []
        
        # Get reference data for the detected pose
        pose_ref = self.pose_references[pose_name]
        
        # Calculate key metrics for the pose
        metrics = self._calculate_pose_metrics(pose_name, landmarks)
        
        # Calculate accuracy score based on metrics
        score, feedback = self._evaluate_pose_accuracy(pose_name, metrics)
        
        return pose_ref['description'], score, feedback
    
    def get_guidance(self, pose_name):
        """
        Get step-by-step guidance for a specific pose
        """
        if pose_name not in self.pose_references:
            return [f"No guidance available for '{pose_name}'"]
        
        return self.pose_references[pose_name]['guidance']
    
    def _calculate_pose_metrics(self, pose_name, landmarks):
        """
        Calculate metrics specific to each pose
        """
        metrics = {}
        landmarks_array = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
        
        if pose_name == 'warrior2':
            # Arms horizontal (shoulder to wrist)
            left_shoulder = landmarks_array[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_wrist = landmarks_array[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_shoulder = landmarks_array[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_wrist = landmarks_array[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            left_arm_angle = self._calculate_angle_to_horizontal(left_shoulder[:2], left_wrist[:2])
            right_arm_angle = self._calculate_angle_to_horizontal(right_shoulder[:2], right_wrist[:2])
            metrics['arms_horizontal'] = abs(left_arm_angle) + abs(right_arm_angle) / 2
            
            # Front knee angle
            right_hip = landmarks_array[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks_array[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks_array[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            front_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            metrics['front_knee_angle'] = front_knee_angle
            
            # Torso upright
            left_hip = landmarks_array[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_shoulder = landmarks_array[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            
            torso_angle = self._calculate_angle_to_vertical(left_hip[:2], left_shoulder[:2])
            metrics['torso_upright'] = abs(torso_angle)
            
            # Back leg straight
            left_hip = landmarks_array[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks_array[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks_array[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            back_leg_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            metrics['back_leg_straight'] = back_leg_angle
            
        elif pose_name == 'tree':
            # Standing leg straight
            left_hip = landmarks_array[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks_array[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks_array[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            standing_leg_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            metrics['standing_leg_straight'] = standing_leg_angle
            
            # Hip openness
            left_hip = landmarks_array[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks_array[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks_array[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            
            hip_angle = self._calculate_angle(left_hip, right_hip, right_knee)
            metrics['hip_openness'] = hip_angle
            
            # Torso upright
            left_hip = landmarks_array[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_shoulder = landmarks_array[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            
            torso_angle = self._calculate_angle_to_vertical(left_hip[:2], left_shoulder[:2])
            metrics['torso_upright'] = abs(torso_angle)
            
            # Foot placement
            left_ankle = landmarks_array[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_knee = landmarks_array[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            
            foot_placement = self._calculate_normalized_distance(left_ankle, right_knee)
            metrics['foot_placement'] = foot_placement * 100  # Scale for readability
            
        # More pose metrics calculations for other poses (plank, goddess, downdog) would follow the same pattern
        # For brevity, we'll just include these two poses as examples
        
        return metrics
    
    def _evaluate_pose_accuracy(self, pose_name, metrics):
        """
        Evaluate the accuracy of the pose based on the calculated metrics
        """
        if pose_name not in self.pose_references:
            return 0.0, []
            
        pose_ref = self.pose_references[pose_name]
        feedback = []
        total_score = 0.0
        total_weight = 0.0
        
        # Evaluate each key point
        for key, ref_values in pose_ref['key_points'].items():
            if key in metrics:
                weight = ref_values.get('weight', 1.0)
                total_weight += weight
                
                if 'target' in ref_values:
                    # For metrics with target values (like angles)
                    target = ref_values['target']
                    threshold = ref_values.get('threshold', 10)
                    deviation = abs(metrics[key] - target)
                    
                    if deviation <= threshold / 2:
                        point_score = 1.0
                    elif deviation <= threshold:
                        point_score = 0.5
                    else:
                        point_score = 0
                        
                    # Generate feedback based on deviation
                    if deviation > threshold:
                        if key == 'front_knee_angle':
                            if metrics[key] < target:
                                feedback.append(f"Bend your front knee more to reach 90 degrees")
                            else:
                                feedback.append(f"Your front knee is bent too much, straighten it slightly")
                        elif key == 'back_leg_straight':
                            feedback.append(f"Keep your back leg straighter")
                        elif 'arm' in key:
                            feedback.append(f"Adjust your arms to be more parallel to the floor")
                        elif 'leg' in key and 'straight' in key:
                            feedback.append(f"Try to straighten your leg more")
                else:
                    # For metrics with thresholds (like deviations from ideal)
                    threshold = ref_values.get('threshold', 10)
                    value = metrics[key]
                    
                    if value <= threshold / 2:
                        point_score = 1.0
                    elif value <= threshold:
                        point_score = 0.5
                    else:
                        point_score = 0
                        
                    # Generate feedback based on value
                    if value > threshold:
                        if key == 'torso_upright':
                            feedback.append(f"Keep your torso more upright")
                        elif 'hip' in key:
                            feedback.append(f"Focus on proper hip alignment")
                        elif 'alignment' in key:
                            part = key.split('_')[0]
                            feedback.append(f"Check your {part} alignment")
                
                total_score += point_score * weight
        
        # Calculate final score as percentage
        if total_weight > 0:
            final_score = (total_score / total_weight) * 100
        else:
            final_score = 0
            
        # If no specific feedback but score is low, give general guidance
        if not feedback and final_score < 70:
            feedback.append(f"Try to follow the guidance for {pose_name} pose more closely")
            
        return final_score, feedback
    
    def _calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points (in degrees)
        """
        a = np.array(a[:2])  # Use only x,y coordinates
        b = np.array(b[:2])
        c = np.array(c[:2])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical errors
        
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        
        return angle
    
    def _calculate_angle_to_horizontal(self, a, b):
        """
        Calculate the angle between a line and the horizontal axis
        """
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def _calculate_angle_to_vertical(self, a, b):
        """
        Calculate the angle between a line and the vertical axis
        """
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        
        angle = np.degrees(np.arctan2(dx, dy))
        
        return angle
    
    def _calculate_normalized_distance(self, a, b):
        """
        Calculate the normalized distance between two points
        """
        a = np.array(a[:2])
        b = np.array(b[:2])
        
        distance = np.linalg.norm(a - b)
        
        # Normalize by the torso length
        torso_length = 1.0  # Ideally, this would be calculated from landmarks
        
        return distance / torso_length 