class Ball:

    def __init__(self, initial_x=0, initial_y=0, initial_xv=0, initial_yv=0, radius=1):
        self.x = initial_x
        self.y = initial_y
        self.xv = initial_xv
        self.yv = initial_yv
        self.radius = radius

    def reset(self, x, y, xv, yv):
        self.x = x
        self.y = y
        self.xv = xv
        self.yv = yv

    def update(self, dt, state):
        self.x += self.xv * dt
        self.y += self.yv * dt
        left_paddle_collision = 0
        right_paddle_collision = 0
        top_field_collision = 0
        bottom_field_collision = 0

        # move to state manager
        if self.y - self.radius <= state.field.top:
            self.yv *= -1
            top_field_collision = 1

        if self.y + self.radius >= state.field.bottom:
            self.yv *= -1  # Reverse vertical velocity
            bottom_field_collision = 1

        # Check for paddle collisions
        if self.x - self.radius < state.left_paddle.x + state.left_paddle.width and self.xv <= 0:  # Left paddle
            if state.left_paddle.y <= self.y <= state.left_paddle.y + state.left_paddle.height:
                self.xv *= -1  # Reverse horizontal velocity
                left_paddle_collision = 1

        if self.x + self.radius > state.right_paddle.x and self.xv > 0:  # Right paddle
            if state.right_paddle.y <= self.y <= state.right_paddle.y + state.right_paddle.height:
                self.xv *= -1  # Reverse horizontal velocity
                right_paddle_collision = 1

        return [left_paddle_collision, right_paddle_collision, top_field_collision, bottom_field_collision]
