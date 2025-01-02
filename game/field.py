class Field:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.left = width / 2.0 * -1
        self.right = width / 2.0
        self.top = height / 2.0 * -1
        self.bottom = height / 2.0
