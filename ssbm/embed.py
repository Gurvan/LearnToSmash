class EmbedPlayer():
    def __init__(self):
        self.n = 10

    def __call__(self, player_state):
        action_state = int(player_state.action_state)
        x = player_state.x / 100.
        y = player_state.y / 100.
        percent = player_state.percent / 100.
        facing = [0, 1] if player_state.facing > 0 else [1, 0]
        jump_used = [0, 1] if bool(player_state.jumps_used) else [1, 0]
        in_air = [0, 1] if player_state.in_air else [1, 0]

        return [
                action_state,
                x, y,
                percent,
               ] + facing + jump_used + in_air

class EmbedGame():
    def __init__(self):
        self.embed_player = EmbedPlayer()
        self.n = self.embed_player.n * 2

    def __call__(self, game_state):
        player_data, opponent_data = [self.embed_player(game_state.players[i]) for i in [0, 1]]
        return player_data + opponent_data


class DummyEmbedGame():
    def __init__(self):
        self.embed_player = EmbedPlayer()
        self.n = self.embed_player.n * 2

    def __call__(self, game_state):
        state = self.n * [0.0]
        return state
