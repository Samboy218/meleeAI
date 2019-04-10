import melee


def act(gamestate, controller):
    shine = False
    for projectile in gamestate.projectiles:
        if abs(projectile.x - gamestate.ai_state.x) < 30 and abs(projectile.y - gamestate.ai_state.y) < 20:
            shine = True
    if shine:
        melee.techskill.multishine(ai_state=gamestate.ai_state, controller=controller)
    else:
        controller.empty_input()

