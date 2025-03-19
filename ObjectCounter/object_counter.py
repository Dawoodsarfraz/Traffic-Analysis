class ObjectCounter:
    def __init__(self):
        """
        Initializes the object counter with a dictionary to store counts per class.
        """
        self.class_counts = {}  # Stores count of objects per class
        self.tracked_ids = set()  # Stores track IDs to ensure unique counting

    def count_object(self, obj):
        """
        Increments the count for an object's class if it has crossed the intrusion line.
        """
        if obj.intrusion_detected and obj.track_id not in self.tracked_ids:
            self.tracked_ids.add(obj.track_id)
            self.class_counts[obj.class_label] = self.class_counts.get(obj.class_label, 0) + 1


    def get_class_counts(self):
        """
        Returns the dictionary containing counts of intruded objects per class.
        """
        return self.class_counts