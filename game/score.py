class Score:
    def __init__(self):
        self.left_score = 0
        self.right_score = 0
        self.left_blocked = 0
        self.right_blocked = 0

    def reset(self):
        self.left_score = 0
        self.right_score = 0
        self.left_blocked = 0
        self.right_blocked = 0

    def update(self, left_score, right_score, left_blocked, right_blocked):
        self.left_score += left_score
        self.right_score += right_score
        self.left_blocked += left_blocked
        self.right_blocked += right_blocked