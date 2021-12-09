import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np

from pokemon_json import *
from poke_common import *
import random
import math
from sklearn.preprocessing import LabelBinarizer
import math




class Actions(enum.Enum):
    Attack_Slot_1 = 0
    Attack_Slot_2 = 1
    Attack_Slot_3 = 2
    Attack_Slot_4 = 3
    Attack_Z_Slot_1 = 4
    Attack_Z_Slot_2 = 5
    Attack_Z_Slot_3 = 6
    Attack_Z_Slot_4 = 7
    Attack_UltraBurst_Slot_1 = 8
    Attack_UltraBurst_Slot_2 = 9
    Attack_UltraBurst_Slot_3 = 10
    Attack_UltraBurst_Slot_4 = 11
    Change_Pokemon_Slot_1 = 12
    Change_Pokemon_Slot_2 = 13
    Change_Pokemon_Slot_3 = 14
    Change_Pokemon_Slot_4 = 15
    Change_Pokemon_Slot_5 = 16
    Change_Pokemon_Slot_6 = 17
    Attack_Struggle = 18



class Player():
    def __init__(self, pokemon):
        self.pokemon = pokemon
        self.curr_pokemon = pokemon[0]
        self.is_player = True

    def encode_pokemon(self, show_full_details=False):

        # if we have empty spots for missing pokemon, just set 0s
        category_encodes = []
        raw_encodes = []
        # add current pokemon first to denote that it is selected.
        for pkmn in self.pokemon:
            cat, raw = self.encode_pokemon_state(pkmn, show_full_details)
            category_encodes.extend(cat)
            raw_encodes.extend(raw)
        return category_encodes, raw_encodes

    def switch_curr_pokemon(self, position):
        new_pokemon = self.pokemon[position]
        self.curr_pokemon = new_pokemon
        return new_pokemon
#Type
#Attack
#SP Attack
# Defense
#SP Defense
# health
# speed
#Ability

    # Bot has some info hidden like atk damage, speed etc. keep hidden to simulate opponent
    def _add_pokemon_encode(self, pokemon):
        # empty pokemon slot
        if pokemon is None and self.is_player:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pokemon is None:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        element_2nd_type = ''
        if pokemon.element_2nd_type != None:
            element_2nd_type = pokemon.element_2nd_type
        item = 'Item Unknown'
        if pokemon.item != None:
            item = pokemon.item
        encode = []
        encode.append(pokemon.name)
        encode.append(pokemon.level)
        encode.append(pokemon.curr_health/float(pokemon.max_health))
        encode.append(pokemon.element_1st_type.value)
        encode.append(element_2nd_type.value)
        encode.append(pokemon.level)
        encode.append(pokemon.weight)

        if self.is_player:
            encode.append(pokemon.item)
            encode.append(pokemon.atk)
            encode.append(pokemon.spatk)
            encode.append(pokemon.defense)
            encode.append(pokemon.spdef)
            encode.append(pokemon.ability)

        return encode

    def encode_attack(self, attack):
        # empty pokemon slot
        if attack is None:
            return ['', '', '', 0, 0, 0, 0]
        raw_encode = [
            attack.num,     # category
            attack.element_type,    # category
            attack.category,  # category
            attack.power,
            attack.accuracy,
            attack.priority,
            attack.pp,
        ]
        return raw_encode

    def encode_pokemon_state(self, pokemon, show_full_details=False):
        # empty pokemon slot
        if pokemon is None and self.is_player:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pokemon is None:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        element_2nd_type = ELEMENT_TYPE.TYPELESS.value
        if pokemon.element_2nd_type != None:
            element_2nd_type = pokemon.element_2nd_type.value
        category_encode = [
            all_pokemon_names_labels.transform([pokemon.num]),     # category
            all_status_labels.transform([pokemon.status.value]),   # category
            all_element_types_labels.transform([pokemon.element_1st_type.value]),     # category
            all_element_types_labels.transform([element_2nd_type]),     # category
            all_abilities_labels.transform([pokemon.ability]),
            all_items_labels.transform([pokemon.item]),
        ]
        raw_encode = []
        if show_full_details:
            raw_encode.append(pokemon.atk)
            raw_encode.append(pokemon.spatk)
            raw_encode.append(pokemon.defense)
            raw_encode.append(pokemon.spdef)

        raw_encode.extend([
            pokemon.level,
            pokemon.curr_health/float(pokemon.max_health),
            pokemon.weight,
            pokemon.accuracy_modifier,
            pokemon.attack_modifier,
            pokemon.spatk_modifier,
            pokemon.defense_modifier,
            pokemon.spdef_modifier,
            pokemon.speed_modifier,
            pokemon.evasion_modifier,
        ])

        attack_category_encode = []
        attack_raw_encode = []
        for attack in pokemon.attacks:
            attack_category_encode.append(all_pokemon_attacks_labels.transform([attack.attack_name]))
            attack_category_encode.append(all_element_types_labels.transform([attack.element_type.value]))
            attack_category_encode.append(all_categories_labels.transform([attack.category.value]))

            attack_raw_encode.append(attack.power)
            accuracy = 1 if attack.accuracy == True else attack.accuracy
            attack_raw_encode.append(accuracy)
            attack_raw_encode.append(attack.priority)
            attack_raw_encode.append(attack.pp)
        category_encode.extend(attack_category_encode)
        raw_encode.extend(attack_raw_encode)

        return category_encode, raw_encode

class RandomAgent(Player):
    def __init__(self, pokemon):
        super().__init__(pokemon)
        self.is_player = False

class State:

    def __init__(self, player, computer_agent):
        self.player = player
        self.computer_agent = computer_agent
        self.weather_condition = 'raindance'
        self.weather_turns = 0
        self.terrain_condition = 'noterrain'
        self.terrain_turns = 0
        self.player_choiced  = False
        self.player_choiced_move = ''
        self.agent_choiced = False
        self.agent_choiced_move = ''
        self.player_selected_pokemon = CurrentPokemon.Pokemon_Slot_1 # category
        self.agent_selected_pokemon  = CurrentPokemon.Pokemon_Slot_1 # category
        self.user_trapped = False
        self.agent_trapped = False
        self.player_used_rocks = False
        self.agent_used_rocks = False
        self.block_next_attack = False
        self.protect_used_last_turn = False
        self.player_has_substitute = False
        self.agent_has_substitute = False
        self.state_transcript = ''
        self.state_reverse_transcript = ''
        #used to switch fainted pokemon without them taking a hit on entry
        self.player_attack_effectiveness = ELEMENT_MODIFIER.NUETRAL  #Category
        self.agent_attack_effectiveness = ELEMENT_MODIFIER.NUETRAL   # Category
        self.player_must_switch = False
        self.agent_must_switch = False
        self.player_flinched = False
        self.agent_flinched = False
        self.player_confused = False
        self.agent_confused = False
        self.player_moved_first = False
        self.player_move_succeeded = False
        self.agent_move_succeeded = False
        self.should_self_print = True
        self.reward = 0
        self.turns = 0

    def reset(self, player, computer_agent, should_print=False):
        self.player = player
        self.computer_agent = computer_agent
        self.turns = 0
        self.should_self_print = should_print
        self.weather_condition = 'raindance'
        self.weather_turns = 0
        self.terrain_condition = 'noterrain'
        self.terrain_turns = 0
        self.player_choiced  = False
        self.player_choiced_move = ''
        self.agent_choiced = False
        self.agent_choiced_move = ''
        self.player_selected_pokemon = CurrentPokemon.Pokemon_Slot_1 # category
        self.agent_selected_pokemon  = CurrentPokemon.Pokemon_Slot_1 # category
        self.user_trapped = False
        self.agent_trapped = False
        self.player_used_rocks = False
        self.agent_used_rocks = False
        self.block_next_attack = False
        self.protect_used_last_turn = False
        self.player_has_substitute = False
        self.agent_has_substitute = False
        self.state_transcript = ''
        self.state_reverse_transcript = ''
        #used to switch fainted pokemon without them taking a hit on entry
        self.player_attack_effectiveness = ELEMENT_MODIFIER.NUETRAL  #Category
        self.agent_attack_effectiveness = ELEMENT_MODIFIER.NUETRAL   # Category
        self.player_must_switch = False
        self.agent_must_switch = False
        self.player_flinched = False
        self.agent_flinched = False
        self.player_confused = False
        self.agent_confused = False
        self.player_moved_first = False
        self.player_move_succeeded = False
        self.agent_move_succeeded = False
        self.reward = 0
        self.turns = 0

    def simulate_battle(self):
        done = False
        winner = None
        steps = 0
        rewards = 0
        while done == False:
            steps += 1
            self.reset_state_transcript()
            print()
            turns_message = 'Turn %d Begin!\n' %(steps,)
            self.printo_magnet(turns_message)
            self.append_to_transcript(turns_message)
            player_action = self.sample_actions(True)
            agent_action = self.sample_actions(False)
            self.printo_magnet(str(player_action))
            self.printo_magnet(str(agent_action))

            self.apply_battle_sequence(player_action, agent_action)
            self.end_of_turn_calc()

            done, winner = self.is_battle_over()
            rewards += self.reward
        self.printo_magnet('winner is: %s' % (winner,))
        return winner, steps, self.reward

    def printo_magnet(self, message):
        message = message.strip()
        if message == '' or self.should_self_print == False:
            return
        player_regex = '_p_'  # for player replace with nothing, for agent replace with opposing
        agent_regex = '_a_'  # for player replace with opposing, for agent replace with nothing
        message = message.replace('_p_', '')
        message = message.replace('_a_', 'Opposing ')

        print(message)

    def end_of_turn_calc(self):
        message = ''
        message = '%s\n%s' % (message, Game.apply_status_damage(self.player.curr_pokemon))
        message = message.strip()
        message = '%s\n%s' % (message, Game.apply_heal_modifiers(self.terrain_condition, self.player.curr_pokemon))
        message = message.strip()
        message = '%s\n%s' % (message, Game.apply_status_damage(self.computer_agent.curr_pokemon))
        message = message.strip()
        message = '%s\n%s' % (message, Game.apply_heal_modifiers(self.terrain_condition, self.computer_agent.curr_pokemon))
        message = message.strip()

        #Lazy reset
        self.player_must_switch = False
        self.agent_must_switch = False
        if self.player.curr_pokemon.curr_health <= 0:
            self.reward -= 20
            self.player_must_switch = True
            message = '_p_%s\n%s has fainted' % (message, self.player.curr_pokemon.name)
            message = message.strip()
        if self.computer_agent.curr_pokemon.curr_health <= 0:
            self.reward += 20
            self.agent_must_switch = True
            message = '_a_%s\n%s has fainted' % (message, self.computer_agent.curr_pokemon.name)
            message = message.strip()

        self.printo_magnet(message)
        self.append_to_transcript(message)

    def encode_field_state(self):
        player_choiced_move = SelectedAttack.Attack_Slot_1.value if self.player_choiced_move == '' else self.player_choiced_move.value
        agent_choiced_move = SelectedAttack.Attack_Slot_1.value if self.agent_choiced_move == '' else self.agent_choiced_move.value
        player_selected_pokemon = '' if self.player_selected_pokemon == '' else self.player_selected_pokemon.value
        agent_selected_pokemon = '' if self.agent_selected_pokemon == '' else self.agent_selected_pokemon.value
        category_encode = [
            all_weather_labels.transform([self.weather_condition]),     # category
            all_terrains_labels.transform([self.terrain_condition]),     # category
            all_attack_slot_labels.transform([player_choiced_move]),   # category
            all_attack_slot_labels.transform([agent_choiced_move]),    # category
#            all_pokemon_slot_labels.transform([player_selected_pokemon]), # category
#            all_pokemon_slot_labels.transform([agent_selected_pokemon]),# category
            0,
            0,
            all_effectiveness_labels.transform([self.player_attack_effectiveness.value]), # category
            all_effectiveness_labels.transform([self.agent_attack_effectiveness.value]),# category
        ]

        raw_encode = [
            self.weather_turns,
            self.terrain_turns,
            self.bool_to_int(self.player_choiced),
            self.bool_to_int(self.agent_choiced),
            self.bool_to_int(self.user_trapped),
            self.bool_to_int(self.agent_trapped),
            self.bool_to_int(self.player_used_rocks),
            self.bool_to_int(self.agent_used_rocks),
            self.bool_to_int(self.player_has_substitute),
            self.bool_to_int(self.agent_has_substitute),
            self.bool_to_int(self.player_must_switch),
            self.bool_to_int(self.player_flinched),
            self.bool_to_int(self.agent_flinched),
            self.bool_to_int(self.player_confused),
            self.bool_to_int(self.agent_confused),
            self.bool_to_int(self.player_moved_first),
            self.bool_to_int(self.player_move_succeeded),
            self.bool_to_int(self.agent_move_succeeded),
        ]

        return category_encode, raw_encode

    def get_observation(self, includeTranscript=False):

        normal_encodes = []
        reverse_encodes = []



        field_cat, field_raw = self.encode_field_state()
        play_cat, play_raw = self.player.encode_pokemon(True)
        age_cat, age_raw = self.computer_agent.encode_pokemon(False)

        normal_encodes.extend(field_cat)
        normal_encodes.extend(play_cat)
        normal_encodes.extend(age_cat)

        # flatten categories
        normal_encodes = flatten(normal_encodes)

        categories_length = len(normal_encodes)

        normal_encodes.extend(field_raw)
        normal_encodes.extend(play_raw)
        normal_encodes.extend(age_raw)

        obs_length = len(normal_encodes)

        #reverse version

        play_cat, play_raw = self.player.encode_pokemon(False)
        age_cat, age_raw = self.computer_agent.encode_pokemon(True)


        reverse_encodes.extend(field_cat)
        reverse_encodes.extend(age_cat)
        reverse_encodes.extend(play_cat)

        # flatten categories
        reverse_encodes = flatten(reverse_encodes)

        reverse_encodes.extend(field_raw)
        reverse_encodes.extend(age_raw)
        reverse_encodes.extend(play_raw)

        if not includeTranscript:
            return normal_encodes
        results = {
            'transcript': self.state_transcript,
            'reverse_transcript': self.state_reverse_transcript,
            'field': self.encode_field_state(),
            'player': self.player.encode_pokemon(),
            'agent':self.computer_agent.encode_pokemon(),
            'combined': normal_encodes,
            'reversed_combined': reverse_encodes,   # used for ai battles
            'cat_length':categories_length,
            'full_obs_len':obs_length,
            'raw_length':(obs_length-categories_length),
            'valid_moves_player': self.get_valid_moves_for_player(),
            'valid_moves_agent': self.get_valid_moves_for_agent(),
            'valid_onehot_player': self.valid_onehot_moves(self.get_valid_moves_for_player()),
            'valid_onehot_agent': self.valid_onehot_moves(self.get_valid_moves_for_agent()),
        }
        return results

    def valid_onehot_moves(self, avail_moves):
        moves = np.zeros(19)
        for move in avail_moves:
            moves[move.value] = 1
#        moves[np.nonzero(moves==0)] = -500
        moves[np.nonzero(moves==0)] = -math.inf
        moves[np.nonzero(moves==0)] = -4000000
        moves[np.nonzero(moves==1)] = 0
        return moves


    def bool_to_int(self, value):
        return 1 if value else 0

    def reset_state_transcript(self):
        self.state_transcript = ''
        self.state_reverse_transcript = ''

    def append_to_transcript(self, message):
        message = message.strip()
        if message == '':
            return
        player_regex = '_p_'  # for player replace with nothing, for agent replace with opposing
        agent_regex = '_a_'  # for player replace with opposing, for agent replace with nothing
        self.state_transcript = '%s\n%s' % (self.state_transcript, message)
        self.state_transcript = self.state_transcript.replace('_p_', '')
        self.state_transcript = self.state_transcript.replace('_a_', 'Opposing ')

        # apply reverse logic.
        self.state_reverse_transcript = '%s\n%s' % (self.state_reverse_transcript, message)
        self.state_reverse_transcript = self.state_reverse_transcript.replace('_a_', '')
        self.state_reverse_transcript = self.state_reverse_transcript.replace('_p_', 'Opposing ')

    def force_switch(self, is_player):
        if is_player:
            available_switches = [pkmn for pkmn in self.player.pokemon if pkmn.curr_health > 0 and pkmn != self.player.curr_pokemon]
            if len(available_switches) > 0:
                self.player.curr_pokemon = available_switches[random.randint(0, len(available_switches)-1)]
                message = '_p_%s was dragged out' % (self.player.curr_pokemon,)
                self.printo_magnet(message)
                self.append_to_transcript(message)
        else:
            available_switches = [pkmn for pkmn in self.computer_agent.pokemon if pkmn.curr_health > 0 and pkmn != self.computer_agent.curr_pokemon]
            if len(available_switches) > 0:
                self.computer_agent.curr_pokemon = available_switches[random.randint(0, len(available_switches)-1)]
                message = '_a_%s was dragged out' % (self.computer_agent.curr_pokemon,)
                self.printo_magnet(message)
                self.append_to_transcript(message)


    def perform_switch(self, switch_action, is_player):
        switch_position = switch_action.value - 12
        old_pokemon = self.player.curr_pokemon if is_player else self.computer_agent.curr_pokemon
        new_pokemon = self.player.switch_curr_pokemon(switch_position) if is_player else self.computer_agent.switch_curr_pokemon(switch_position)
        # convert change to switch position
        if is_player:
            self.player_selected_pokemon = CurrentPokemon(switch_action.value)
        else:
            self.agent_selected_pokemon = CurrentPokemon(switch_action.value)
        self.reset_stats_on_entry(new_pokemon, is_player)
        message = 'Come back %s!, Let\'s bring it %s!' % (old_pokemon.name, new_pokemon.name)
        self.printo_magnet(message)
        self.append_to_transcript(message)
        self.apply_entry_hazard(new_pokemon, is_player)

    def apply_painsplit(self):
        total_hp = self.player.curr_pokemon.curr_health + self.computer_agent.curr_pokemon.curr_health
        self.player.curr_pokemon.curr_health = min(self.player.curr_pokemon.max_health, math.ceil(total_hp/2.0))
        self.computer_agent.curr_pokemon.curr_health = min(self.computer_agent.curr_pokemon.max_health, math.ceil(total_hp/2.0))

    def perform_attack_sequence(self, act, pkmn, enemy, is_player=False):
        attack = pkmn.attacks[act.value]
        accuracy = attack.accuracy
        message = "%s used %s" % (pkmn.name, attack.attack_name)
        self.printo_magnet(message)
        self.append_to_transcript(message)
        attack.pp -= 1
        self.printo_magnet('accuracy: %.2f'% (accuracy,))
        if not (self.block_next_attack and attack.target != TARGET.SELF) and (accuracy is not True or abs(random.random()) < accuracy):
            if is_player:
                self.player_move_succeeded = True
            else:
                self.agent_move_succeeded = True

            # Handle status differently. or do NOTHING
            if attack.category == CATEGORY.STATUS:
                status = attack.status
                if status != None:
                    tar = enemy if attack.target == TARGET.NORMAL else pkmn
                    self.apply_boosts(status, True, tar)
            else:
                #duplicate code to determine if super effective or resisted
                modifier = 1
                modifier *= Game.effective_modifier(enemy.element_1st_type, attack)
                modifier *= Game.effective_modifier(enemy.element_2nd_type, attack)
                message = ''
                effective_reward = 0
                if modifier > 1:
                    message = 'It was super effective'
                    if is_player:
                        effective_reward = 5
                    else:
                        effective_reward = -5
                elif modifier == 0:
                    message = 'It doesn\'t effect %s' % (enemy.name)
                    if is_player:
                        effective_reward = -5
                    else:
                        effective_reward = 5
                elif modifier < 1:
                    message = 'It\'s not very effective'
                    effective_reward = -5
                    if is_player:
                        effective_reward = -2
                    else:
                        effective_reward = 5
                if message != '':
                    self.printo_magnet(message)
                    self.append_to_transcript(message)
                self.reward += effective_reward

                # calculate recoil damage based on health change
                attack_damage = self.calculate(attack, pkmn, enemy)
                damage_dealt = enemy.curr_health-attack_damage
                if is_player and self.player_has_substitute:
                    recoil_damage_pokemon = math.ceil(min(self.player_substitue_health, damage_dealt)*0.33)
                    self.player_substitue_health -= attack_damage
                    if self.player_substitue_health <= 0:
                        self.player_has_substitute = False
                        message = '_p_Substitute broke!'
                        self.printo_magnet(message)
                        self.append_to_transcript(message)
                elif not is_player and self.agent_has_substitute:
                    recoil_damage_pokemon = math.ceil(min(self.agent_substitue_health, damage_dealt)*0.33)
                    self.agent_substitue_health -= attack_damage
                    if self.agent_substitue_health <= 0:
                        self.agent_has_substitute = False
                        message = '_a_Substitute broke!'
                        self.printo_magnet(message)
                        self.append_to_transcript(message)
                else:
                    health_lost = abs(enemy.curr_health-damage_dealt)
                    if health_lost > enemy.curr_health:
                        health_lost = enemy.curr_health
                    recoil_damage_pokemon = math.ceil(min(enemy.curr_health, abs(damage_dealt))*0.33)
                    message = "%s lost %.2f health" % (enemy.name, health_lost)
                    enemy.curr_health = max(0, damage_dealt)
                    self.printo_magnet(message)
                    self.append_to_transcript(message)
                if attack.has_recoil:
                    pkmn.curr_health -= recoil_damage_pokemon
                    message = "%s lost %.2f health from recoil" % (pkmn.name, recoil_damage_pokemon)
                    self.printo_magnet(message)
                    self.append_to_transcript(message)


            # stealth rocks special conditions
            if attack.id == 'stealthrock':
                message = 'Jagged rocks thrown on field'
                self.printo_magnet(message)
                self.append_to_transcript(message)
                if is_player:
                    self.player_used_rocks = True
                else:
                    self.agent_used_rocks = True

            if attack.id == 'defog' or attack.attack_name == 'rapidspin':
                # reverse for removing rocks
                if is_player:
                    self.agent_used_rocks = False
                    message = 'Rocks removed from field'
                    self.printo_magnet(message)
                    self.append_to_transcript(message)
                else:
                    self.player_used_rocks = False
                    message = 'Oppsing team removed rocks'
                    self.printo_magnet(message)
                    self.append_to_transcript(message)

            if attack.id in ['softboiled', 'recover', 'roost', 'morningsun']:
                pkmn.curr_health = min(pkmn.curr_health + pkmn.max_health/2, pkmn.max_health)
                message = "%s restored some health" % (pkmn.name,)
                self.printo_magnet(message)
                self.append_to_transcript(message)

            if attack.id == 'healbell':
                self.heal_all_partners(is_player)

            if attack.id == 'roar':
                self.force_switch(is_player)


            if attack.id == 'painsplit':
                self.apply_painsplit()
                message1 = '%s shared the pain' % pkmn.name
                message2 = '%s shared the pain' % enemy.name
                self.printo_magnet(message1)
                self.append_to_transcript(message1)
                self.printo_magnet(message2)
                self.append_to_transcript(message2)

            if attack.id == 'protect' and not self.protect_used_last_turn:
                self.block_next_attack = True

            # apply boosts and effects/
            if attack.boosts is not None and len(attack.boosts) > 0:
                for boost in attack.boosts:
                    if isinstance(attack.boosts, dict):
                        boost = (boost, attack.boosts[boost])
                    tar = enemy if attack.target == TARGET.NORMAL else pkmn
                    self.apply_boosts(boost, True, tar)

            twenty_five_percent = pkmn.max_health / 4.0
            if attack.id == 'substitute' and pkmn.curr_health > twenty_five_percent:
                pkmn.curr_health -= twenty_five_percent
                if is_player:
                    self.player_has_substitute = True
                    self.player_substitue_health = twenty_five_percent
                else:
                    self.agent_has_substitute = True
                    self.agent_substitue_health = twenty_five_percent

            # apply secondary
            secondary = attack.secondary
            if secondary is not None:
                # apply boosts and effects/
                if secondary.boosts is not None and len(secondary.boosts) > 0:
                    for boost in secondary.boosts:
                        tar = enemy if attack.target == TARGET.NORMAL else pkmn
                        self.apply_boosts(boost, True, tar)

                if secondary.status is not None:
                    tar = pkmn if secondary.is_target_self  else enemy
                    self.apply_boosts(secondary.status, secondary.chance, tar)
        else:
            message = 'But it failed/Missed'
            if is_player:
                self.player_move_succeeded = False
            else:
                self.agent_move_succeeded = False
            self.printo_magnet(message)
            self.append_to_transcript(message)
            self.block_next_attack = False

    def heal_all_partners(self, is_player):
        if is_player:
            for pkmn in self.player.pokemon:
                if pkmn.status != Status.NOTHING:
                    pkmn.status = Status.NOTHING
                    message = '%s is cured' % (pkmn.name,)
                    self.printo_magnet(message)
                    self.append_to_transcript(message)
        else:
            for pkmn in self.computer_agent.pokemon:
                if pkmn.status != Status.NOTHING:
                    pkmn.status = Status.NOTHING
                    message = '%s is cured' % (pkmn.name,)
                    self.printo_magnet(message)
                    self.append_to_transcript(message)

    def apply_boosts(self, boost, chance, target):
        if chance is not True and random.random() > chance:
            return

        if boost == 'par':
            target.status = Status.PARALYSIS
            message = '%s is paralyzed' % (target.name,)
            self.printo_magnet(message)
            self.append_to_transcript(message)
            return
        if boost == 'brn':
            target.status = Status.BURN
            message = '%s is burned' % (target.name,)
            self.printo_magnet(message)
            self.append_to_transcript(message)
            return

        if boost == 'psn':
            target.status = Status.POISON
            message = '%s is poisoned' % (target.name,)
            self.printo_magnet(message)
            self.append_to_transcript(message)
            return
        if boost == 'tox':
            target.status = Status.TOXIC
            message = '%s is badly poisoned' % (target.name,)
            self.printo_magnet(message)
            self.append_to_transcript(message)
            return

        # perhaps an unimplemented status early exit
        if not isinstance(boost, tuple):
            return

        stat, val = boost[0], boost[1]
        modifier_value = 1.5 if val > 0 else 0.5
        incre_decre = 'increased' if val > 0 else 'decreased'
        if stat == 'atk':
            target.attack_modifier *= modifier_value
        if stat == 'def':
            target.defense_modifier *= modifier_value
        if stat == 'spa':
            target.spatk_modifier *= modifier_value
        if stat == 'spd':
            target.spdef_modifier *= modifier_value
        if stat == 'spe':
            target.speed_modifier *= modifier_value
        if stat == 'evasion':
            target.evasion_modifier *= modifier_value
        if stat == 'accuracy':
            target.accuracy_modifier *= modifier_value
        message = '%s\'s %s has %s'% (target.name, stat, incre_decre)
        self.printo_magnet(message)
        self.append_to_transcript(message)

    def reset_stats_on_entry(self, pokemon, is_player=True):
        pokemon.accuracy_modifier = 1
        pokemon.attack_modifier = 1
        pokemon.spatk_modifier = 1
        pokemon.defense_modifier = 1
        pokemon.spdef_modifier = 1
        pokemon.speed_modifier = 1
        pokemon.accuracy_modifier = 1
        message = Game.apply_natural_cure(pokemon)
        self.printo_magnet(message)
        self.append_to_transcript(message)

        # reset choice
        if is_player:
            self.player_choiced = False
            self.player_has_substitute = False
            self.player_substitue_health = 0
            self.player_choiced_move = ''
        else:
            self.agent_choiced = False
            self.agent_has_substitute = False
            self.agent_substitue_health = 0
            self.agent_choiced_move = ''

    def apply_entry_hazard(self, pokemon, is_player=True):
        if is_player and self.agent_used_rocks:
            # remove 25 % health
            pokemon.curr_health = max(0, pokemon.curr_health-(pokemon.max_health/4))
            message = 'Rocks dugged into _p_ %s' % (pokemon.name,)
            self.printo_magnet(message)
            self.append_to_transcript(message)
        if is_player == False and self.player_used_rocks:
            # remove 25 % health
            pokemon.curr_health = max(0, pokemon.curr_health-(pokemon.max_health/4))
            message = 'Rocks dugged into _a_ %s' % (pokemon.name,)
            self.printo_magnet(message)
            self.append_to_transcript(message)

    def sample_actions(self, is_player=False):
        if is_player:
            actions = self.get_valid_moves_for_player()
        else:
            actions = self.get_valid_moves_for_agent()
        return np.random.choice(actions, 1)[0]

    def take_action_if_valid_else_random_actions(self, action):
        actions = self.get_valid_moves_for_player()
        if action in actions:
            return action

        return np.random.choice(actions, 1)[0]

    def is_battle_over(self):
        player_living_pokemon_count = len([pkmn for pkmn in self.player.pokemon if pkmn.curr_health > 0])
        agent_living_pokemon_count = len([pkmn for pkmn in self.computer_agent.pokemon if pkmn.curr_health > 0])

        if player_living_pokemon_count == 0:
            self.reward -= 100
            return True, 'agent'
        if agent_living_pokemon_count == 0:
            self.reward += 100
            return True, 'player'

        return False, None

    def get_valid_moves_for_player(self):
        curr_pokemon = self.player.curr_pokemon
        pokemon = self.player.pokemon
        valid_moves  = []
        if curr_pokemon.curr_health > 0:
            if self.player_choiced:
                if curr_pokemon.attacks[self.player_choiced_move].pp > 0:
                    valid_moves.append(Actions(self.player_choiced_move))

            else:
                for idx, atk in enumerate(curr_pokemon.attacks):
                    if atk is not None and atk.pp > 0:
                        valid_moves.append(Actions(idx))

            if len(valid_moves) == 0: # can only struggle
                valid_moves.append(Actions.Attack_Struggle)


        for idx, pkmn in enumerate(self.player.pokemon):
            if pkmn.curr_health > 0 and pkmn is not curr_pokemon:
                valid_moves.append(Actions(12+idx))
        return valid_moves

    def get_valid_moves_for_agent(self):
        curr_pokemon = self.computer_agent.curr_pokemon
        pokemon = self.computer_agent.pokemon
        valid_moves  = []
        if curr_pokemon.curr_health > 0:
            if self.agent_choiced:
                if curr_pokemon.attacks[self.agent_choiced_move].pp > 0:
                    valid_moves.append(Actions(self.player_choiced_move))

            else:
                for idx, atk in enumerate(curr_pokemon.attacks):
                    if atk is not None and atk.pp > 0:
                        valid_moves.append(Actions(idx))

            if len(valid_moves) == 0: # can only struggle
                valid_moves.append(Actions.Attack_Struggle)


        for idx, pkmn in enumerate(self.computer_agent.pokemon):
            if pkmn.curr_health > 0 and pkmn is not curr_pokemon:
                valid_moves.append(Actions(12+idx))
        return valid_moves

    # dont handle u turn but it is good to think about.
    def apply_battle_sequence(self, action, computer_action):
        assert isinstance(action, Actions)
        assert isinstance(computer_action, Actions)
        # reset
        self.player_attack_effectiveness = ELEMENT_MODIFIER.NUETRAL  #Category
        self.agent_attack_effectiveness = ELEMENT_MODIFIER.NUETRAL   # Category

        player_is_first, actions_order = Game.get_action_order(self.player, self.computer_agent, action, computer_action)

        # When pokemon faint, only valid switches are allowed.
        if self.player_must_switch or self.agent_must_switch:
            if self.player_must_switch and action in [Actions.Change_Pokemon_Slot_1, Actions.Change_Pokemon_Slot_2, Actions.Change_Pokemon_Slot_3, Actions.Change_Pokemon_Slot_4, Actions.Change_Pokemon_Slot_5, Actions.Change_Pokemon_Slot_6]:
                self.perform_switch(action, True)
            if self.agent_must_switch and computer_action in [Actions.Change_Pokemon_Slot_1, Actions.Change_Pokemon_Slot_2, Actions.Change_Pokemon_Slot_3, Actions.Change_Pokemon_Slot_4, Actions.Change_Pokemon_Slot_5, Actions.Change_Pokemon_Slot_6]:
                self.perform_switch(computer_action, False)
            #dont allow any fighting
            return

        self.player_moved_first = player_is_first

        for idx, act in enumerate(actions_order):
            # perform here in case switch was engaged
            player_pokemon = self.player.curr_pokemon
            agent_pokemon = self.computer_agent.curr_pokemon

            if idx == 0:
                is_player = player_is_first
            else:
                is_player = not player_is_first

            if is_player:
                pkmn = player_pokemon
                enemy = agent_pokemon
            else:
                pkmn = agent_pokemon
                enemy = player_pokemon

            # Handle switch
            if act in [Actions.Change_Pokemon_Slot_1, Actions.Change_Pokemon_Slot_2, Actions.Change_Pokemon_Slot_3, Actions.Change_Pokemon_Slot_4, Actions.Change_Pokemon_Slot_5, Actions.Change_Pokemon_Slot_6]:
                self.perform_switch(act, is_player)
                self.reward -= 2

            # if pokemon fainted, cant attack
            if pkmn.curr_health <= 0:
                continue

            if act in [Actions.Attack_Slot_1, Actions.Attack_Slot_2, Actions.Attack_Slot_3, Actions.Attack_Slot_4]:
                self.perform_attack_sequence(act, pkmn, enemy, is_player)
                # if contains a choice, lock attack
                if pkmn.item == ITEMS.CHOICE_SCARF:
                    if is_player:
                        self.player_choiced = True
                        self.player_choiced_move = act
                    else:
                        self.agent_choiced = True
                        self.agent_choiced_move = act



            # struggle deals 10% damage to target, 25% to user. Keep it simple
            if act == Actions.Attack_Struggle:
                enemy.curr_health = max(0, enemy.curr_health-(enemy.max_health/10))
                pkmn.curr_health = min(0, pkmn.curr_health-(pkmn.max_health/4))
                message = '%s used struggle ' % (pkmn.name,)
                self.printo_magnet(message)
                self.append_to_transcript(message)

        #reset protect
        if self.protect_used_last_turn:
            self.protect_used_last_turn = False

        if self.block_next_attack:
            self.block_next_attack = False
            self.protect_used_last_turn = True


    def calculate(self, attack, pkmn, target):
        base_atk = pkmn.atk if attack.category == CATEGORY.PHYSICAL else pkmn.spatk
        base_def = target.defense if attack.category == CATEGORY.PHYSICAL else target.spdef

        return (((2*pkmn.level/5+2) * attack.power * (base_atk/base_def))/50 + 2) * self.get_modifier(pkmn, target, attack)

    def get_modifier(self, pokemon, enemy, attack):
        modifier = 1.5 if self.is_stab(pokemon, attack) else 1
        modifier *= Game.apply_attack_modifiers(attack, pokemon, enemy, self.weather_condition, self.terrain_condition)
        message = 'modifier is: %.2f' % (modifier,)
        self.printo_magnet(message)
        self.append_to_transcript(message)
        return modifier

    def is_stab(self, pokemon, attack):
        return pokemon.element_1st_type == attack.element_type or pokemon.element_2nd_type == attack.element_type


    @property
    def shape(self):
        # player has 6 pokemon with 16 slots
        # computer agent has 6 pokemon with 11 slots since some info hidden.
#        return (len(self.encode()['combined']), )
        return (len(self.encode()), )

    def encode(self, includeTranscript=False):
        """
        Convert current state into numpy array.
        """
        return self.get_observation(includeTranscript)

    def step(self, player_action, agent_action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(player_action, Actions)
        self.reward = 0
        self.reset_state_transcript()

        self.turns += 1
        turns_message = 'Turn %d Begin!\n' %(self.turns,)
        self.printo_magnet(turns_message)
        self.append_to_transcript(turns_message)
        self.printo_magnet(str(player_action))
        self.printo_magnet(str(agent_action))


        self.apply_battle_sequence(player_action, agent_action)
        self.end_of_turn_calc()

        done, winner = self.is_battle_over()

        # Penalty for taking too long
        if self.turns >= 150:
            done = True
            self.reward = -500
            winner = 'agent'

        return self.reward, done, winner


class PokeEnv(gym.Env):
#class PokeEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self, player, computer_agent, computer_network=None):
        # if network exists, use that decide moves. Otherwise use random move.
        self._state = State(player, computer_agent)
        self.player = player
        self.computer_agent = computer_agent
        self.computer_network = computer_network

        self.action_space = gym.spaces.Discrete(n=len(Actions))
#        self.observation_space = gym.spaces.Discrete(n=self._state.shape[0])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, self._state.shape[0]), dtype=np.float32)
        self.seed()

    def set_network(computer_network):
        self.computer_network = computer_network

    def sample_actions(self):
        return self._state.sample_actions(True)

    def reset(self):
        self.player.pokemon = get_random_pokemon_team()
        self.computer_agent.pokemon = get_random_pokemon_team()
        self._state.reset(self.player, self.computer_agent)
        return self._state.encode()

    def get_current_transcript(self):
        return self._state.state_transcript

    def step(self, action_idx):
        action = Actions(action_idx)
        if self.computer_network == None:
            agent_action = self._state.sample_actions(False)
        else:
            obs = self._state.encode(includeTranscript=True)
            agent_transcript = obs['reverse_transcript']
            agent_obs = obs['reversed_combined']
            agent_valid_moves = obs['valid_onehot_agent']
            # Get the action
            action, value, _ = computer_network.step(obs, [valid_moves], [transcript])

        newaction = self._state.take_action_if_valid_else_random_actions(action)
        if action != newaction:
            print('Bot chose %s but got %s' % (action, newaction))
            obs = self._state.encode(includeTranscript=True)
            print('prev: valid_onehot', obs['valid_onehot_player'])

        reward, done, winner = self._state.step(newaction, agent_action)
        obs = self._state.encode(includeTranscript=True)
        info = {"transcript":obs['transcript'], "valid_onehot_player": obs['valid_onehot_player'],
         'cat_length':obs['cat_length'], 'full_obs_len':obs['full_obs_len'], 'raw_length':obs['raw_length'],
         'winner': winner}
#        print('valid_onehot', obs['valid_onehot_player'])
        return obs['combined'], reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

class RewardScaler(gym.RewardWrapper):
    """  may not be needed for fake pokemon?
        Bring rewards to a reasonable scale for PPO.
        This is incrediably important and effects performance drastically
    """
    def reward(self, reward):
        return reward * 0.01


class Game(object):

    @staticmethod
    def get_action_order(player, computer_agent, action, computer_action):
        actions_order = []
        # switches happen first
        if computer_action in [Actions.Change_Pokemon_Slot_1, Actions.Change_Pokemon_Slot_2, Actions.Change_Pokemon_Slot_3, Actions.Change_Pokemon_Slot_4, Actions.Change_Pokemon_Slot_5, Actions.Change_Pokemon_Slot_6]:
            actions_order.append(computer_action)
            actions_order.append(action)
            return False, actions_order

        if action in [Actions.Change_Pokemon_Slot_1, Actions.Change_Pokemon_Slot_2, Actions.Change_Pokemon_Slot_3, Actions.Change_Pokemon_Slot_4, Actions.Change_Pokemon_Slot_5, Actions.Change_Pokemon_Slot_6]:
            actions_order.append(action)
            actions_order.append(computer_action)
            return True, actions_order

        # Handle struggle case
        if action == Actions.Attack_Struggle:
            actions_order.append(computer_action)
            actions_order.append(action)
            return False, actions_order

        if computer_action == Actions.Attack_Struggle:
            actions_order.append(action)
            actions_order.append(computer_action)
            return True, actions_order


        player_pokemon = player.curr_pokemon
        agent_pokemon = computer_agent.curr_pokemon
#        print('computer action', computer_action.value)
#        print('player action', action.value)
        # check priority
        if agent_pokemon.attacks[computer_action.value].priority > player_pokemon.attacks[action.value].priority:
            actions_order.append(computer_action)
            actions_order.append(action)
            return False, actions_order

        if agent_pokemon.attacks[computer_action.value].priority < player_pokemon.attacks[action.value].priority:
            actions_order.append(action)
            actions_order.append(computer_action)
            return True, actions_order

        # check pokemon speed
        if agent_pokemon.speed * Game.apply_speed_modifier(agent_pokemon) > player_pokemon.speed * Game.apply_speed_modifier(player_pokemon):
            actions_order.append(computer_action)
            actions_order.append(action)
            return False, actions_order

        if agent_pokemon.speed * Game.apply_speed_modifier(agent_pokemon) < player_pokemon.speed * Game.apply_speed_modifier(player_pokemon):
            actions_order.append(action)
            actions_order.append(computer_action)
            return True, actions_order

        # pick randomly
        if random.random() > 0.5:
            actions_order.append(computer_action)
            actions_order.append(action)
            return False, actions_order
        else:
            actions_order.append(action)
            actions_order.append(computer_action)
            return True, actions_order

    @staticmethod
    def apply_attack_modifiers(attack, attacker, defender, weather, terrain):
        modifier = 1
        modifier *= Game.effective_modifier(defender.element_1st_type, attack)
        modifier *= Game.effective_modifier(defender.element_2nd_type, attack)
        modifier *= Game.weather_modifier(weather, attack.element_type)
        modifier *= Game.apply_burn_modifier(attacker, attack)
        modifier *= Game.terrain_modifier(terrain, attack.element_type)
        modifier *= Game.apply_levitate_modifier(attack.element_type, defender.ability)
        modifier *= Game.pure_power_modifier(attacker, attack)
        modifier *= Game.big_fist_modifier(attacker)
        return modifier

    @staticmethod
    def effective_modifier(defender, attack):
        if defender is None:
            return 1
        element_modifier = get_damage_modifier_for_type(defender, attack)
        return element_modifier


    @staticmethod
    def apply_sandstorm_modifier(weather, defender):
        # rock types get 1.5x SPDEF in sand
        if weather == WEATHER.SANDSTORM and (ELEMENT_TYPE.ROCK == defender.element_1st_type or ELEMENT_TYPE.ROCK == defender.element_2nd_type):
            return 1.5
        return 1

    @staticmethod
    def weather_modifier(weather, attack_element):
        if weather == WEATHER.SUN and (ELEMENT_TYPE.FIRE == attack_element):
            return 1.5
        if weather == WEATHER.RAIN and (ELEMENT_TYPE.WATER == attack_element):
            return 1.5
        return 1

    @staticmethod
    def apply_burn_modifier(pokemon, attack):
        if Status.BURN == pokemon.status and attack.category == CATEGORY.PHYSICAL:
            return 0.5
        return 1

    @staticmethod
    def terrain_modifier(terrain, attack_element):
        if terrain == TERRAIN.ELECTRIC_TERRAIN and (ELEMENT_TYPE.ELECTRIC == attack_element):
            return 1.5
        if terrain == TERRAIN.GRASSY_TERRAIN and (ELEMENT_TYPE.GRASS == attack_element):
            return 1.5
        if terrain == TERRAIN.MISTY_TERRAIN and (ELEMENT_TYPE.FAIRY == attack_element):
            return 1.5
        if terrain == TERRAIN.PSYCHIC_TERRAIN and (ELEMENT_TYPE.PSYCHIC == attack_element):
            return 1.5
        return 1

    @staticmethod
    def apply_levitate_modifier(attack_element, ability):
        if ELEMENT_TYPE.GROUND == attack_element and ('Levitate' == ability):
            return 0
        return 1


    @staticmethod
    def pure_power_modifier(pokemon, attack):
        if Ability.PURE_POWER == pokemon.ability and attack.category == CATEGORY.PHYSICAL:
            return 2
        return 1

    @staticmethod
    def big_fist_modifier(pokemon):
        if Ability.BIG_FIST == pokemon.ability:
            return 1.5
        return 1

    @staticmethod
    def apply_speed_modifier(pokemon):
        speed_modifier = 1
        if ITEMS.CHOICE_SCARF == pokemon.item:
            speed_modifier *= 1.5
        if Status.PARALYSIS == pokemon.status:
            speed_modifier *= .5
        return 1

    # end of turn modifiers
    @staticmethod
    def apply_natural_cure(pokemon):
        message = ''
        if ('Natural Cure' == pokemon.ability) and pokemon.status != Status.NOTHING:
            message = '%s is cured' % (pokemon.name,)
        return message

    @staticmethod
    def apply_status_damage(pokemon):
        message = ''
        if Status.BURN == pokemon.status:
            pokemon.curr_health -= 6
            message = '%s was hurt by burn' % (pokemon.name,)
        if Status.POISON == pokemon.status or Status.TOXIC == pokemon.status:
            pokemon.curr_health -= 12
            message = '%s was hurt by poison' % (pokemon.name,)
        return message

    @staticmethod
    def apply_heal_modifiers(terrain, pokemon):
        message = ''
        if pokemon.curr_health <=0:
            return
        if terrain == TERRAIN.GRASSY_TERRAIN:
            pokemon.curr_health = min(pokemon.max_health, pokemon.curr_health+(pokemon.max_health/12))
            message += '%s recovered a little hp with grassy terrain\n' % (pokemon.name,)
        if ITEMS.LEFT_OVERS == pokemon.item:
            pokemon.curr_health = min(pokemon.max_health, pokemon.curr_health+(pokemon.max_health/12))
            message += '%s recovered a little hp with leftovers' % (pokemon.name,)
        return message


def make_env(player, computer_agent, computer_network=None):
    env = PokeEnv(player, computer_agent, computer_network)
    env = RewardScaler(env)

    return env

def make_poke_env():
    def _init():
        team1 = get_random_pokemon_team()
        team2 = get_random_pokemon_team()
        player = Player(team1)
        random_agent = RandomAgent(team2)

        env = make_env(player, random_agent)
        return env
    return _init
