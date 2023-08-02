import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import pygame




class State:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.value = 0
        # self.directions = {'(1,0)': 0,'(-1,0)': 0, '(0, 1)': 0, '(0, -1)': 0}
        # self.max_direction = max(self.directions, key=self.directions.get)

    def update_direction(self, key, value):
        self.directions[key] = value

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self) -> str:
        return str(self.x) + ',' + str(self.y)
    

class Action:
    def __init__(self, dx: int, dy: int):
        self.dx = dx
        self.dy = dy

    def __eq__(self, other) -> bool:
        return self.dx == other.dx and self.dy == other.dy

    def __hash__(self) -> int:
        return hash((self.dx, self.dy))

    def __str__(self) -> str:
        return f'({self.dx}, {self.dy})'


class ModelTR:
    def __init__(self):
        self.T = {}
        self.R = {}

    def update(self, state: State, action: Action, r: float, next_state: State):
        self.T[(state, action)] = next_state
        self.R[(state, action)] = r

    def sample(self):
        state, action = list(self.T)[rnd.randint(0, len(self.T.keys()) - 1)]
        next_state = self.T[(state, action)]
        r = self.R[(state, action)]
        return state, action, r, next_state


class LearnMemory:
    def update(state: State, action: Action, q: float):
        pass

    def Q(state: State, action: Action) -> float:
        pass


class SparseMemory(LearnMemory):
    def __init__(self, default: float = 0):
        self.default = default
        self.memory = {}

    def Q(self, state: State, action: Action) -> float:

        act =  self.memory.get((state, action), self.default)
        # print(self.memory," ecolhe:->", state.x, state.y, action.dx, action.dy)
        # exit()
        return act

    def update(self, state: State, action: Action, q: float):
        self.memory[(state, action)] = q

    def dim(self):
        return len(self.memory)

    def dictS(self):
        return self.memory
    
    def __str__(self):
        return [(s.__str__(), a.__str__(), v) for (s, a), v in self.memory.items()]


class ActionSelection:
    def select_action(self, state: State) -> Action:
        pass


class EGreedy(ActionSelection):
    def __init__(self, memory: LearnMemory, actions: list, epsilon: float):
        self.memory = memory
        self.actions = actions
        self.epsilon = epsilon

    def max_action(self, state: State) -> Action:
        rnd.shuffle(self.actions)
        # print("actions:", [a.__str__() for a in self.actions], " -> ", m)
        # print(key=lambda a: self.memory.Q(state, a))
        # print(m)
        # return m
        return max(self.actions, key=lambda a: self.memory.Q(state, a))
        
    def exploit(self, state: State) -> Action:
        return self.max_action(state)

    def explore(self) -> Action:
        return self.actions[rnd.randint(0, len(self.actions) - 1)]
        
    def select_action(self, state: State) -> Action:

        if rnd.random() > self.epsilon: #0.1
            # print("maxim ation:" , self.exploit(state), "state:", state,  self.epsilon)
            return self.exploit(state)
        else:
            # print("Eplore:", state)
            maxi = self.explore()
            # print("maxim ation", maxi)
            return maxi 
            


class ReinforcedLearning:
    def __init__(self, memory: LearnMemory, action_selection: ActionSelection, alpha: float, gama: float):
        self.memory = memory
        self.action_selection = action_selection
        self.alpha = alpha
        self.gama = gama

    def learn(state: State, action: Action, r: float, next_state: State, next_action: Action = None):
        pass


class QLearning(ReinforcedLearning):
    def learn(self, state: State, action: Action, r: float, next_state: State):
        next_action = self.action_selection.max_action(next_state)
        q_state_action = self.memory.Q(state, action)
        q_nest_state_next_action = self.memory.Q(next_state, next_action)
        q = q_state_action + self.alpha * (r + self.gama * q_nest_state_next_action - q_state_action)
        # print(self.alpha, self.gama)
        self.memory.update(state, action, q)
        # print("q:", q, "state:", state, "action:", action, "r:", r, "next_state:", next_state, "next_action:", next_action)


class Experience():
    def __init__(self, state: State, action: Action, r: float, next_state: State):
        self.state= state
        self.action= action
        self.r = r
        self.next_state = next_state

# MEMÓRIA DE EXPERIÊNCIA (QME)
class MemoriaExperience():
    def __init__(self, dim_max):
        self.dim_max = dim_max
        self.memory = []
    
    def update(self, e): #atualizar()
        if len(self.memory) == self.dim_max:
            self.memory.remove(self.memory[0]) # remover o primeiro elemento
        self.memory.append(e)

    def samples(self, n): # Amostrar()
        n_samples = min(n, len(self.memory))
        return rnd.sample(self.memory, n_samples)

# Q-LEARNING COM MEMÓRIA DE EXPERIÊNCIA (QME) dim_max = ?
class QWE(QLearning):
    def __init__(self, memory: LearnMemory, action_selection: ActionSelection, alpha: float, gama: float, sim_num: int, dim_max: int):
        super().__init__(memory, action_selection, alpha, gama)
        self.sim_num = sim_num
        self.memory = memory
        self.experienceMemory = MemoriaExperience(dim_max)

    def learn(self, state: State, action: Action, r: float, next_state: State):
        super().learn(state, action, r, next_state)
        e = (state, action, r, next_state)
        self.experienceMemory.update(e)
        self.simulate()
    
    def simulate(self):
        samples = self.experienceMemory.samples(self.sim_num)
        for (state, action, r, next_state) in samples:
            super().learn(state, action, r, next_state)


class DynaQ(QLearning):
    def __init__(self, memory: LearnMemory, action_selection: ActionSelection, alpha: float, gama: float, sim_num: int):
        super().__init__(memory, action_selection, alpha, gama)
        self.sim_num = sim_num
        self.model = ModelTR()

    def simulate(self):
        for i in range(self.sim_num):
            state, action, r, next_state = self.model.sample()
            super().learn(state, action, r, next_state)

    def learn(self, state: State, action: Action, r: float, next_state: State):
        super().learn(state, action, r, next_state)
        self.model.update(state, action, r, next_state)
        self.simulate()


class World:
    def __init__(self, file_name: str, multiplier: float = 10, 
    move_cost: float = 0.01, showGraf: bool = True, algorithm_name="Unknom"):
    
        self.file_name = file_name
        self.world, self.state, self.target = self.loadworld()
        self.MemoryWorld = [['0' for x in y] for y in self.world] # np.zeros(self.world.shape)

        self.algorithm_name = algorithm_name
        self.showGraf = showGraf
        self.multiplier = multiplier
        self.move_cost = move_cost
        self.movements = 0
        self.turn = 1

    def loadworld(self):
        with open(self.file_name, "r") as file:
            lines = file.readlines()
            world: list[list[int]] = np.zeros((len(lines), len(lines[0].replace('\n', ''))), dtype=int)

            initial_state = State(0, 0)
            target = State(0, 0)

            # Find obstacles [O], initial position [>] and target position [A]:
            m = 0
            for x in lines:
                n = 0
                for y in x[:-1]:
                    if y.__eq__('O'):
                        world[m][n] = -1
                    elif y.__eq__('A'):
                        world[m][n] = 2
                        target = State(n, m)
                    elif y.__eq__('>'):
                        # world[m][n] = 1
                        initial_state = State(n, m)
                    n += 1
                m += 1

            # print(world)
            # print('initial_state:', initial_state)
            # print('target:', target)
            return world, initial_state, target

    def current_state(self) -> State:
        return self.state

    def update_state(self, next_state: State, sc, TILE):
        # print("self.state={}, next_state={}".format(self.state, next_state))
        self.state = next_state
        
    def printW(self):
        matrix = []
        for i in range(len(self.world)):
            l = []
            for j in range(len(self.world[0])):
                l.append(self.world[j][i].getValue())
            matrix.append(l)
            #print(l)
        plt.imshow(matrix)
        plt.show()
        
    def move(self, action: Action):
        next_state = State(self.state.x + action.dx, self.state.y + action.dy)

        # Increment movements:
        self.movements += 1
        # Next position:
        r = self.world[next_state.y][next_state.x]
        # print(next_state, "valor:", r)
        # print("r =", r, self.state)
        # Retun next state and reinforcement, acording to position and move cost:
        # print("ns= \n", next_state) # (0,5)
        # print(r) # -1
        # print(self.state) #(1,5)
        # print(self.multiplier) # 10
        # print(self.move_cost) #0.01
        # if r >= 0:
        #     print(self.state, r * self.multiplier - self.move_cost)
        # print("r=", r)
        return next_state if r >= 0 else self.state, r * self.multiplier - self.move_cost

    def iter_states(self): # passa de matrix para lista
        position = [[x for x in y] for y in self.world]
        position[self.state.y][self.state.x] = 1  # Colocar o agente na posicao atual

        # self.world[self.state.y][self.state.x].setValue(1)
        # world[n][m] = State(n, m)
        # world[n][m].setValue(1)
        for i in range(len(self.world)):
            for j in range(len(self.world[0])):
                yield self.world[i][j]

    
    def drawState(self, sc, TILE, x, y, value, valorEstado_direction):
        x, y = x * TILE, y * TILE        
        
        l = 1

        if value == -1: # Wall
            pygame.draw.rect(sc, pygame.Color('dodgerblue4'), (x + l, y + l, TILE-l, TILE-l)) # navy gray20
        if value == 2: # Target
            pygame.draw.rect(sc, (0, 98, 2), (x + l, y + l, TILE-l, TILE-l)) # red3 brown4 gold2
        if value == 1: # Agent
            pygame.draw.rect(sc, pygame.Color('yellow1'), (x + l, y + l, TILE-l, TILE-l)) # aquamarine4 cadetblue4 mediumorchid4 dodgerblue4
        if value == 0: # background
            pygame.draw.rect(sc, pygame.Color('azure3'), (x + l, y + l, TILE-l, TILE-l))
                        
            if valorEstado_direction != '0':
                (valorEstado, action) = valorEstado_direction
                # print(type(action))

                # pintar a celula -6 red florescente, <0 red fraco, >0 green fraco, > 8 green forte
                v = float(valorEstado)
                if  v < -6:
                    # print(int(float(valorEstado)))
                    pygame.draw.rect(sc, (140, 0, 0), (x + l, y + l, TILE-l, TILE-l))
                if -6 < v < 0:
                    pygame.draw.rect(sc, (230, 0, 0), (x + l, y + l, TILE-l, TILE-l))
                if 8 > v > 0 :
                    pygame.draw.rect(sc, pygame.Color('limegreen'), (x + l, y + l, TILE-l, TILE-l))
                if v > 8:
                    pygame.draw.rect(sc, pygame.Color('lawngreen'), (x + l, y + l, TILE-l, TILE-l)) # limegreen
                # mostrar o valor da celula
                # font = pygame.font.SysFont('Calibri', 8, True, False)
                # v = '%.4f'%(float(valorEstado))
                # text = font.render(v, True, (0, 0, 0))
                
                # direcao da seta
                
                setaD = pygame.image.load("arrow/arrowD.png")
                # print("action", action)
                if action.dx == 1 and action.dy == 0: # right
                    setaD = pygame.transform.rotate(setaD, 90)
                    # text = pygame.transform.rotate(text, 90)
                    
                if action.dx == -1 and action.dy == 0: #action == (-1,0): #left
                    setaD = pygame.transform.rotate(setaD, -90)
                    # text = pygame.transform.rotate(text, -90)
                if action.dx == 0 and action.dy == 1: #action == (0, 1):
                    setaD = pygame.transform.rotate(setaD, 0)
                    # text = pygame.transform.rotate(text, 0)
                if action.dx == 0 and action.dy == -1: #action == (0, -1):
                    # print("direcao: left", Action(1,0))
                    setaD = pygame.transform.rotate(setaD, 180)
                    # text = pygame.transform.rotate(text, 180)

                sc.blit(setaD, [x, y])
                #Atualizar o dictionario de self.dictions
                # sc.blit(text, [x+5, y+5])

    def show(self, sc, TILE, agentMemory):
        position = [[x for x in y] for y in self.world]
        position[self.state.y][self.state.x] = 1  # Colocar o agente na posicao atual

        bestAction = {}
        for (s, a), v in list(agentMemory.items()):
            if (s.x, s.y) in bestAction:
                if v > bestAction[(s.x, s.y)][0]:
                    bestAction[(s.x, s.y)] = (v, a)
            else:
                bestAction[(s.x, s.y)] = (v, a)
        
        for x in range(len(position)):
            for y in range(len(position[x])):
                self.drawState(sc, TILE, x, y, position[y][x], bestAction.get((x, y), '0'))
        



        # print(self.MemoryWorld)
    def regenerate(self):
        self.movements = 0
        self.turn += 1
        self.world, self.state, self.target = self.loadworld()


# MecanismoAprendRef
class ReinforcementLearningMechanism:
    def __init__(self, actions):
        self.actions = actions
        self.memory = SparseMemory()
        self.action_selection = EGreedy(self.memory, self.actions, 0.1)


class Agent():
    def __init__(self, actions, method_name):
        self.actions = actions
        self.mecanism = ReinforcementLearningMechanism(self.actions)
        self.end = False
        self.learning = self.learningMethod(method_name)
        
    def learningMethod(self, method_name):
        if method_name == "QLearning":
            # print("method_name:", method_name)
            self.learning = QLearning(self.mecanism.memory, self.mecanism.action_selection, 0.7, 0.95)
            return self.learning
        if method_name == "DynaQ": # simula todas as acoes possiveis
            self.learning = DynaQ(self.mecanism.memory, self.mecanism.action_selection, 0.7, 0.95, 10)
            return self.learning
        if method_name == "QME": # Q-LEARNING COM MEMÓRIA DE EXPERIÊNCIA - escolhe e refaz
            self.learning = QWE(self.mecanism.memory, self.mecanism.action_selection, 0.7, 0.95, 10, 3)
            return self.learning
        
        else:
            print("Metodo errado-->", method_name)
        

        

class PerformanceEvaluation:
    # classe para avaliar o desempenho do agente, com
    # métricas como a taxa de sucesso, 
    # a recompensa total obtida, 
    # o número de iterações necessárias
    # a convergência do algoritmo
    def __init__(self, num_episodes, threshold):
        self.threshold = threshold # 6
        self.num_episodes = num_episodes # 10
        self.Rewards = []
        self.total_reward = 0
        self.thresholdConvergence = 0.9

        self.success = 0
        self.iterations = 0
        self.convergence = 0
        # self.convergence_list = []

    def episode(self, reward):
        self.Rewards.append(reward)
        self.iterations += 1
        if reward >= self.threshold:
            self.success += 1
        if self.success_rate(self.Rewards, self.threshold) >= self.thresholdConvergence:
            self.convergence = self.iterations
            self.convergence_list.append(self.iterations)
            # print("Convergiu em: ", self.convergence)
    
    def episode_start(self):
        self.total_reward = 0
    
    def episode_end(self):
        self.episode(self.total_reward)
    
    def sucessRate(self):
        return self.success / self.iterations
    
    def totalReward(self):
        return self.total_reward
    
    def iterations(self):
        return self.iterations

    def convergence(self):
        return self.convergence
    
    def success_rate(self, rewards, threshold):
        if len(rewards) == 0:
            return 0
        successful_episodes = 0
        for r in rewards:
            if r >= threshold:
                successful_episodes += 1
        return successful_episodes / len(rewards)
    
    def show_results(self):
        print("Taxa de sucesso: ", self.success_rate(self.Rewards, self.threshold))
        print("Recompensa total: ", self.total_reward)
        print("Número de iterações: ", self.iterations)
        print("Convergência: ", self.convergence)
        print("Lista Rewards: ", self.Rewards)
    
    def save_results(self, file):
        file.write("Taxa de sucesso: " + str(self.success_rate(self.Rewards, self.threshold)) + " ")
        file.write("Recompensa total: " + str(self.total_reward) + " ")
        file.write("Número de iterações: " + str(self.iterations) + " ")
        file.write("Convergência: " + str(self.convergence) + " ")
        file.write("Lista Convergência: " + str(self.convergence_list) + " ")
        file.write("Lista Rewards: " + str(self.Rewards) + " ")


    def reset(self):
        self.success = 0
        self.total_reward = 0
        self.iterations = 0
        self.convergence = 0
        self.convergence_list = []
    

    