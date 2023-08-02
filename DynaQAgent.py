
from classes import *
import sys
import time

if len(sys.argv) < 3:
    print("Usage: python DynaQAgent.py algorithm_name['QME', 'DynaQ', 'QLearning'] amb['amb1.txt', 'amb2.txt']")
    sys.exit(1)
algorithm_name = sys.argv[1]
amb = sys.argv[2]
# print("algorithm_name: ", algorithm_name, amb)

actions = [
    Action(1, 0),  # Right
    Action(-1, 0),  # Left
    Action(0, -1),  # Up
    Action(0, 1)  # Down
]



# algorithm_name='DynaQ' # QME' DynaQ QLearning

# Create world:
world = World(amb, algorithm_name=algorithm_name)


# create agent with actions
agent = Agent(actions=actions, method_name=algorithm_name)

# create classe PerformanceEvaluation
performance = PerformanceEvaluation(num_episodes=100, threshold=6)
# performance.num_episodes = 10

# Start process
if amb == "amb1.txt":
    RES = WIDTH, HEIGHT = 602, 602
    TILE = 50    # Tamanho da celula

if amb == "amb2.txt":
    RES = WIDTH, HEIGHT = 680, 680
    TILE = 30    # Tamanho da celula

pygame.init()
sc = pygame.display.set_mode(RES)
clock = pygame.time.Clock()
pygame.display.set_caption("Agente Inteligente")

sair = False
c = 1

def run():
    while True:
        sc.fill(pygame.Color('black')) #darkslategray
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        

        # Agente com o estado seleciona a acao 
        action = agent.mecanism.action_selection.select_action(world.current_state())
        # Mover o agente
        next_state, r = world.move(action)

        agent.learningMethod(algorithm_name)
        agent.learning.learn(world.current_state(), action, r, next_state)


        world.update_state(next_state, sc, TILE)

        # print(" -4- agent.mecanism.memory.memory: ")
        # Mostrar o mundo
        world.show(sc, TILE, agent.mecanism.memory.memory)
        if (next_state == world.target):

            print(" -5- Agente chegou ao objetivo")

            # Carrega o mundo, colocando o agente de volta ao início
            world.regenerate()
            # performance.episode_start()
            return r, 
            
        # pygame.display.flip()
        pygame.time.wait(10)
        pygame.display.update()
        clock.tick(500)




# Loop for episodes
for episode in range(performance.num_episodes):
    print("Episode: ", episode)
    # Carrega o mundo, colocando o agente de volta ao início
    performance.episode_start()
    run()
    if episode == performance.num_episodes-1:
        time.sleep(1000)

    performance.episode_end()

performance.show_results()
# performance.save_results()


        





