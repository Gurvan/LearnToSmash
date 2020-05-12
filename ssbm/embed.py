class EmbedPlayer():
    def __init__(self):
        self.n = 8

    def __call__(self, player_state):
        action_state = int(player_state.action_state)
        x = player_state.x / 100.
        y = player_state.y / 100.
        percent = player_state.percent / 100.
        shield_size = player_state.shield_size / 60.0
        facing = int(player_state.facing > 0)
        jump_used = int(player_state.jumps_used > 0)
        in_air = int(player_state.in_air)

        return [
                action_state,
                x, y,
                percent,
                shield_size,
                facing,
                jump_used,
                in_air
               ]

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
