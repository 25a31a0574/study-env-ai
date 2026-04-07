class StudyEnv:
    def __init__(self, level="easy"):
        self.level = level
        self.reset()

    def reset(self):
        if self.level == "easy":
            self.time = 6
            self.tasks = ["Math", "Python", "Physics"]
        elif self.level == "medium":
            self.time = 8
            self.tasks = ["Math", "Python", "Physics", "DSA", "English"]
        else:
            self.time = 10
            self.tasks = ["Math", "Python", "Physics", "DSA", "English", "AI"]
            self.energy = 5

        self.done = []
        return self.get_state()

    def get_state(self):
        state = {
            "time_left": self.time,
            "remaining_tasks": self.tasks
        }
        if self.level == "hard":
            state["energy"] = self.energy
        return state

    def step(self, action):
        reward = 0

        if action in self.tasks:
            self.tasks.remove(action)
            self.done.append(action)
            reward = 1
        elif action == "Break":
            reward = -0.5
            if self.level == "hard":
                self.energy += 1
        else:
            reward = -1

        self.time -= 1

        if self.level == "hard":
            self.energy -= 1

        done = self.time <= 0 or len(self.tasks) == 0

        return self.get_state(), reward, done
