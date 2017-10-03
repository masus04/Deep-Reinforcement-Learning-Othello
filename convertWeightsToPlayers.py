import src.config as config
import src.player as player

pl1 = player.TDPlayer(config.BLACK)
pl2 = player.TDPlayer(config.WHITE)

for plr in (pl1, pl2):
    plr.load_params()
    plr.save()
