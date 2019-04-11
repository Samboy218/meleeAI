#!/usr/bin/python3
import melee
import argparse
import signal
import sys
import testAi

#This example program demonstrates how to use the Melee API to run dolphin programatically,
#   setup controllers, and send button presses over to dolphin

def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
         raise argparse.ArgumentTypeError("%s is an invalid controller port. \
         Must be 1, 2, 3, or 4." % value)
    return ivalue

chain = None

parser = argparse.ArgumentParser(description='Example of libmelee in action')
parser.add_argument('--port', '-p', type=check_port,
                    help='The controller port your AI will play on',
                    default=2)
parser.add_argument('--opponent', '-o', type=check_port,
                    help='The controller port the opponent will play on',
                    default=1)
parser.add_argument('--live', '-l',
                    help='The opponent is playing live with a GCN Adapter',
                    default=True)
parser.add_argument('--debug', '-d', action='store_true',
                    help='Debug mode. Creates a CSV of all game state')
parser.add_argument('--framerecord', '-r', default=False, action='store_true',
                    help='(DEVELOPMENT ONLY) Records frame data from the match, stores into framedata.csv.')
parser.add_argument('--bot', '-b',
                    help='Opponent is a second instance of SmashBot', action='store_true')

args = parser.parse_args()
print(args.bot)

log = None
if args.debug:
    log = melee.logger.Logger()

framedata = melee.framedata.FrameData(args.framerecord)

#Options here are:
#   "Standard" input is what dolphin calls the type of input that we use
#       for named pipe (bot) input
#   GCN_ADAPTER will use your WiiU adapter for live human-controlled play
#   UNPLUGGED is pretty obvious what it means
opponent_type = melee.enums.ControllerType.UNPLUGGED
if args.live and not args.bot:
    opponent_type = melee.enums.ControllerType.GCN_ADAPTER
if args.bot:
    opponent_type = melee.enums.ControllerType.STANDARD

#Create our Dolphin object. This will be the primary object that we will interface with
dolphin = melee.dolphin.Dolphin(ai_port=args.port,
                                opponent_port=args.opponent,
                                opponent_type=opponent_type,
                                logger=log)
#Create our GameState object for the dolphin instance
gamestate = melee.gamestate.GameState(dolphin)
#Create our Controller object that we can press buttons on
controller = melee.controller.Controller(port=args.port, dolphin=dolphin)
controller2 = melee.controller.Controller(port=args.opponent, dolphin=dolphin)

def signal_handler(signal, frame):
    dolphin.terminate()
    if args.debug:
        log.writelog()
        print("") #because the ^C will be on the terminal
        print("Log file created: " + log.filename)
    print("Shutting down cleanly...")
    if args.framerecord:
        framedata.saverecording()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#Run dolphin and render the output
dolphin.run(render=True)

#Plug our controller in
#   Due to how named pipes work, this has to come AFTER running dolphin
#   NOTE: If you're loading a movie file, don't connect the controller,
#   dolphin will hang waiting for input and never receive it
controller.connect()
if args.bot:
    controller2.connect()

#MY CODE
actor = testAi.BasicLearner(2, 1000)
paused = -1
#Main loop
while True:
    #"step" to the next frame
    gamestate.step()
    if(gamestate.processingtime * 1000 > 12):
        print("WARNING: Last frame took " + str(gamestate.processingtime*1000) + "ms to process.")

    #What menu are we in?
    if gamestate.menu_state in [melee.enums.Menu.IN_GAME]:
        if args.framerecord:
            framedata.recordframe(gamestate)
        #XXX: This is where your AI does all of its stuff!
        #This line will get hit once per frame, so here is where you read
        #   in the gamestate and decide what buttons to push on the controller
        #print("State:", gamestate.tolist())
        #print("projectiles:", [x.tolist() for x in gamestate.projectiles])

        #restart the game if either percent gets to 999 or we're in sudden death
        if gamestate.ai_state.percent >= 999 or gamestate.opponent_state.percent >= 999:
            seq = [melee.enums.Button.BUTTON_L,melee.enums.Button.BUTTON_R,melee.enums.Button.BUTTON_A,melee.enums.Button.BUTTON_START]
            if paused < 0:
                paused = 0
                controller2.press_button(melee.enums.Button.BUTTON_START)
            else:
                #lraS
                if paused > 100:
                    for button in seq:
                        controller2.press_button(button)
                    paused = -1
                else:
                    paused += 1
        else:

            action = actor.doit(gamestate)
            if action == 1:
                controller.press_button(melee.enums.Button.BUTTON_R)
                #If standing, shine
                #if gamestate.ai_state.action == melee.enums.Action.STANDING or gamestate.ai_state.action == melee.enums.Action.KNEE_BEND:
                    #controller.press_button(melee.enums.Button.BUTTON_B)
                    #controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, .5, 0)
                #else:
                    #controller.empty_input()
            else:
                controller.empty_input()

            if args.bot:
                if gamestate.opponent_state.action == melee.enums.Action.STANDING:
                    controller2.press_button(melee.enums.Button.BUTTON_B)
                else:
                    controller2.empty_input()
    #If we're at the character select screen, choose our character
    elif gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
        #print("in char select")
        melee.menuhelper.choosecharacter(character=melee.enums.Character.FOX,
                                        gamestate=gamestate,
                                        port=args.port,
                                        opponent_port=args.opponent,
                                        controller=controller,
                                        swag=True,
                                        start=False)
        if args.bot:
            melee.menuhelper.choosecharacter(character=melee.enums.Character.FOX,
                                             gamestate=gamestate,
                                             port=args.opponent,
                                             opponent_port=args.port,
                                             controller=controller2,
                                             swag=True,
                                             start=True)
    #If we're at the postgame scores screen, spam START
    elif gamestate.menu_state == melee.enums.Menu.POSTGAME_SCORES:
        paused = -1
        melee.menuhelper.skippostgame(controller=controller)
        if args.bot:
            melee.menuhelper.skippostgame(controller=controller2)
    #If we're at the stage select screen, choose a stage
    elif gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
        melee.menuhelper.choosestage(stage=melee.enums.Stage.FINAL_DESTINATION,
                                    gamestate=gamestate,
                                    controller=controller)
    #Flush any button presses queued up
    controller.flush()
    if args.bot:
        controller2.flush()
    if log:
        log.logframe(gamestate)
        log.writeframe()
