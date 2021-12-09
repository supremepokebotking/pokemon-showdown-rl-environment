#Type
#Attack
#SP Attack
# Defense
#SP Defense
# health
# speed
#Ability

import enum
import json
from pokemon_json import *
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import re
from collections import deque

class Action(enum.Enum):
  Attack_Slot_1 = 0
  Attack_Slot_2 = 1
  Attack_Slot_3 = 2
  Attack_Slot_4 = 3
  Attack_Z_Slot_1 = 4
  Attack_Z_Slot_2 = 5
  Attack_Z_Slot_3 = 6
  Attack_Z_Slot_4 = 7
  Attack_Mega_Slot_1 = 8
  Attack_Mega_Slot_2 = 9
  Attack_Mega_Slot_3 = 10
  Attack_Mega_Slot_4 = 11
  Attack_Ultra_Slot_1 = 12
  Attack_Ultra_Slot_2 = 13
  Attack_Ultra_Slot_3 = 14
  Attack_Ultra_Slot_4 = 15
  Change_Slot_1 = 16
  Change_Slot_2 = 17
  Change_Slot_3 = 18
  Change_Slot_4 = 19
  Change_Slot_5 = 20
  Change_Slot_6 = 21
  Attack_Struggle = 22
  Shift_Left = 23           # Triples Only
  Shift_Right = 24          # Triples Only
  Not_Decided = 25          # position hasn't been decided yet


class Ability(enum.Enum):
    LEVITATE = 1
    ILLUSION = 2
    PRANKSTER = 3
    PURE_POWER = 4
    HARVEST = 5
    NATURAL_CURE = 6
    BIG_FIST = 7

class CurrentPokemon(enum.Enum):
    Pokemon_Slot_1 = 12
    Pokemon_Slot_2 = 13
    Pokemon_Slot_3 = 14
    Pokemon_Slot_4 = 15
    Pokemon_Slot_5 = 16
    Pokemon_Slot_6 = 17

class SelectedAttack(enum.Enum):
    Attack_Slot_1 = 1
    Attack_Slot_2 = 2
    Attack_Slot_3 = 3
    Attack_Slot_4 = 4


class GameType(enum.Enum):
    SINGLES = 1
    DOUBLES = 2
    TRIPLES = 3

class GEN(enum.Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8

class ITEMS(enum.Enum):
    BLUE_BERRY = 1
    CHOICE_SCARF = 2
    LEFT_OVERS = 3
    TOXIC_ORB = 5
    WHITE_HERB = 6
    Z_STONE = 7

class VOLATILE_STATUS(enum.Enum):
    NOTHING = 'Nothing'
    CONFUSION = 'Confusion'

class GENDER(enum.Enum):
    UNKNOWN = ''
    MALE = 'M'
    FEMALE = 'F'


class Status(enum.Enum):
    NOTHING = ''
    BURN = 'brn'
    SLEEP = 'Sleep'
    FROZEN = 'Frozen'
    PARALYSIS = 'Paralysis'
    POISON = 'psn'
    TOXIC = 'tox'
    FAINTED = 'fnt'

class WEATHER(enum.Enum):
    NONE = ''
    SUN = ''
    RAIN = 'RainDance'
    HARSH_SUNLIGHT = ''
    DOWNPOUR = ''
    HAIL = 'Hail'
    SANDSTORM = ''

class TERRAIN(enum.Enum):
    NO_TERRAIN = 'noterrain'
    ELECTRIC_TERRAIN = 'electricterrain'
    GRASSY_TERRAIN = 'grassyterrain'
    MISTY_TERRAIN = 'mistyterrain'
    PSYCHIC_TERRAIN = 'psychicterrain'


#Triples       Doubles     Singles
# 3  2  1         2  1         1
#-1 -2 -3        -1 -2        -1


class SELECTABLE_TARGET(enum.Enum):
    DO_NOT_SPECIFY=0,  # Used for most options, singles/random/normal/self,multi... and shifts
    SELF=1,
    FOE_SLOT_1=2,
    FOE_SLOT_2=3,
    FOE_SLOT_3=4,  # Only in triples
    ALLY_SLOT_1=5,
    ALLY_SLOT_2=6,
    ALLY_SLOT_3=7, # Only in triples


class TARGET(enum.Enum):
    #These all use target DO_NOT_SPECIFY
    SELF = 'self'
    ALL_ADJACENT_FOES = 'allAdjacentFoes'
    ALLY_SIDE = 'allySide'
    ALLY_TEAM = 'allyTeam'
    FOE_SIDE = 'foeSide'
    ALL = 'all'
    ALL_ADJACENT = 'allAdjacent'
    RANDOM_NORMAL = 'randomNormal'   # outrage

#require a pick
    NORMAL = 'normal'
    ANY = 'any'
    ADJACENT_FOE = 'adjacentFoe'
    ADJACENT_ALLY = 'adjacentAlly'
    ADJACENT_ALLY_OR_SELF = 'adjacentAllyOrSelf'
    SCRIPTED = 'scripted'

class ELEMENT_TYPE(enum.Enum):
    BUG = "Bug"
    DARK = "Dark"
    DRAGON = "Dragon"
    ELECTRIC = "Electric"
    FAIRLY = "Fairy"
    FIGHTING = "Fighting"
    FIRE = "Fire"
    FLYING = "Flying"
    GHOST = "Ghost"
    GRASS = "Grass"
    GROUND = "Ground"
    ICE = "Ice"
    NORMAL = "Normal"
    POISON = "Poison"
    PSYCHIC = "Psychic"
    ROCK = "Rock"
    STEEL = "Steel"
    WATER = "Water"
    TYPELESS = "Typeless"
    BIRD = "Bird"       # For missing No?

class ELEMENT_MODIFIER(enum.Enum):
    NUETRAL = 0
    SUPER_EFFECTIVE = 1
    RESISTED = 2
    IMMUNE = 3

def get_damage_modifier_for_type(target_pokemon_element, attack):
    damage_modifier = 1
    element_modifier = get_damage_taken(target_pokemon_element, attack.element_type.value)
    if element_modifier == ELEMENT_MODIFIER.NUETRAL:
        damage_modifier = 1
    if element_modifier == ELEMENT_MODIFIER.SUPER_EFFECTIVE:
        damage_modifier = 2
    if element_modifier == ELEMENT_MODIFIER.RESISTED:
        damage_modifier = 0.5
    if element_modifier == ELEMENT_MODIFIER.IMMUNE:
        damage_modifier = 0

    return damage_modifier




"""
    second param is string in case someone wants to test against things like
    paralysis, prankster etc. Not just elements
"""
def get_damage_taken(element_type, to_test_against_name):
    modifier = ELEMENT_MODIFIER.NUETRAL
    element_damage_map = damage_taken_dict[element_type.value]
    if to_test_against_name in element_damage_map:
        element_damage_map = ELEMENT_MODIFIER(element_damage_map[to_test_against_name])
    return element_damage_map


class CATEGORY(enum.Enum):
    STATUS = 'Status'
    PHYSICAL = 'Physical'
    SPECIAL = 'Special'

class Secondary():
    def __init__(self, chance='100', boosts=[], status=None, volatileStatus=None, is_target_self=False):
        self.chance = float(chance) / 100
        self.boosts = boosts
        self.status = status
        self.volatileStatus = volatileStatus
        self.is_target_self = is_target_self

class Attack():
    def __init__(self):
        self.id = ''
        self.attack_name = ''
        self.target = 'normal'
        self.pp  = 0
        self.used_pp = 0
        self.element_type = None
        self.power = 0
        self.accuracy = 0
        self.status = None
        self.category = None
        self.priority = 1
        self.disabled = False
        self.isZ = False


class Pokemon():
    def __init__(self):
        self.name = ''
        #Only used for sending to neural network
        self.form = ''
        self.is_hidden = True
        self.level  = 0
        self.max_health = 1
        self.curr_health = 1
        self.atk = 0
        self.spatk = 0
        self.defense = 0
        self.spdef = 0
        self.speed = 0
        self.weight = 0
        self.ability = ''
        self.element_1st_type = None
        self.element_2nd_type = None
        self.attacks = None
        self.accuracy_modifier = 1
        self.attack_modifier = 1
        self.spatk_modifier = 1
        self.defense_modifier = 1
        self.spdef_modifier = 1
        self.speed_modifier = 1
        self.evasion_modifier = 1
        self.status = None
        self.item = ''
        self.gender = GENDER.UNKNOWN
        self.is_active = False
        self.canMegaEvolve = False
        self.canUltra = False

    def get_hidden_info(self):
        self.name = ''
        self.is_hidden = True
        self.level  = 0
        self.max_health = 1
        self.curr_health = 1
        self.atk = atk
        self.spatk = spatk
        self.defense = defense
        self.spdef = spdef
        self.speed = speed
        self.weight = weight
        self.ability = ability
        self.element_1st_type = element_1st_type
        self.element_2nd_type = element_2nd_type
        self.attacks = attacks
        self.status = Status.NOTHING
        self.item = ''


# 'side_conditions': {'stealthrock': 0, 'spikes': 0, 'toxic_spikes':0}, 'trapped': False, attack_locked:False,
# Attack_Slot_1_disabled:False, Attack_Slot_2_disabled:False, Attack_Slot_4_disabled:False, Attack_Slot_4_disabled:False, },
# 'weather': None, 'terrain': None, 'forceSwitch': False, 'wait': False}

class Ability(enum.Enum):
    LEVITATE = 1
    ILLUSION = 2
    PRANKSTER = 3
    PURE_POWER = 4
    HARVEST = 5
    NATURAL_CURE = 6
    BIG_FIST = 7

class ITEMS(enum.Enum):
    BLUE_BERRY = 1
    CHOICE_SCARF = 2
    LEFT_OVERS = 3
    TOXIC_ORB = 5
    WHITE_HERB = 6
    Z_STONE = 7

def pokemon_from_json(pokemon_data, attacks=None):
    name = pokemon_data['species']
    num = pokemon_data['num']
    # for one hot encoding of pokemon
    all_pokemon_names.add(name)
    level  = 5
    health = pokemon_data['baseStats']['hp']
    atk = pokemon_data['baseStats']['atk']
    spatk = pokemon_data['baseStats']['spa']
    defense = pokemon_data['baseStats']['def']
    spdef = pokemon_data['baseStats']['spd']
    speed = pokemon_data['baseStats']['spe']
    weight = pokemon_data['weightkg']
    ability = pokemon_data['abilities']['0']
    element_1st_type = ELEMENT_TYPE(pokemon_data['types'][0])
    element_2nd_type = None
    if len(pokemon_data['types']) > 1:
        element_2nd_type = ELEMENT_TYPE(pokemon_data['types'][1])

    pokemon = Pokemon()
    pokemon.name = name
    pokemon.level  = level
    pokemon.max_health = 1
    pokemon.curr_health = 1
    pokemon.atk = atk
    pokemon.spatk = spatk
    pokemon.defense = defense
    pokemon.spdef = spdef
    pokemon.speed = speed
    pokemon.weight = weight
    pokemon.ability = ability
    pokemon.element_1st_type = element_1st_type
    pokemon.element_2nd_type = element_2nd_type
    pokemon.attacks = deque([hidden_attack(), hidden_attack(), hidden_attack(), hidden_attack()], maxlen=4)
    pokemon.status = pokemon.status
    pokemon.gender = pokemon.gender
    pokemon.item = pokemon.item
    return pokemon

def attacks_from_json(attack_data, key=None):
    id = attack_data['id']
    # Dont mix num and strings hidden power ruins this
    if 'num' not in attack_data or True:
        num = key
    else:
        num = attack_data['num']
    all_pokemon_attacks.add(num)
    basePower = attack_data['basePower']
    category = CATEGORY(attack_data['category'])
    accuracy = attack_data['accuracy']
    if accuracy is not True:
        accuracy = accuracy / 100.0
    name = attack_data['name']
    pp = attack_data['pp']
    element_type = ELEMENT_TYPE(attack_data['type'])
    ignoreImmunity = False
    if 'ignoreImmunity' in attack_data:
        ignoreImmunity = True
    status = None
    if 'status' in attack_data:
        status = attack_data['status']
    priority = attack_data['priority']
    is_zmove = 'isZ' in attack_data
    target = TARGET(attack_data['target'])
    boosts = None
    volatileStatus = None
    has_recoil = True if 'recoil' in attack_data else False
    if 'boosts' in attack_data:
        boosts = attack_data['boosts']
    if 'volatileStatus' in attack_data:
        volatileStatus = attack_data['volatileStatus']
    secondary = None
    if 'secondary' in attack_data and attack_data['secondary'] is not False:
        sec_boosts = []
        status = None
        is_target_self = False
        if 'boosts' in attack_data['secondary']:
            sec_boosts = attack_data['secondary']['boosts']
        if 'status' in attack_data['secondary']:
            status = attack_data['secondary']['status']

        if 'self' in attack_data['secondary']:
            is_target_self = True
            if 'boosts' in attack_data['secondary']['self']:
                sec_boosts = attack_data['secondary']['self']['boosts']
            if 'status' in attack_data['secondary']['self']:
                status = attack_data['secondary']['self']['status']

        secondary = Secondary(attack_data['secondary']['chance'], sec_boosts, status, volatileStatus, is_target_self)

    isZ = False
    if 'isZ' in attack_data:
        isZ = True


    target = 'normal'
    if 'target' in attack_data:
        target = attack_data['target']

    flags = []
    flag_keys = attack_data['flags'].keys()
    for key in flag_keys:
        flags.append((key, attack_data['flags'][key]))

    attack = Attack()
    attack.id = id
    attack.attack_name = name
    attack.pp  = pp
    attack.element_type = element_type
    attack.power = basePower
    attack.accuracy = accuracy
    attack.status = status
    attack.category = category
    attack.priority = priority
    attack.target = target
    attack.isZ = isZ
    return attack


all_items_by_name = {}
all_items_by_key = {}
all_abilities_by_name = {}
all_abilities_by_key = {}
all_attacks_by_name = {}
all_attacks_by_key = {}


#configured by adding pokemon
all_items = set()
all_generations = set()
all_gametypes = set()
all_tiers = set()
all_genders = set()
all_pokemon_attacks = set()
all_abilities = set()
all_pokemon_names = set()
all_weather = set()
all_status = set()
all_element_types = set()
all_terrains = set()
all_targets = set()
all_selectable_targets = set()
all_categories = set()
all_effectiveness = set()
all_pokemon_slots = set()
all_attack_slots = set()
all_actions = set()
all_rooms = set()

all_items_labels = None
all_generations_labels = None
all_gametypes_labels = None
all_tiers_labels = None
all_genders_labels = None
all_pokemon_attacks_labels = None
all_abilities_labels = None
all_pokemon_names_labels = None
all_weather_labels = None
all_status_labels = None
all_element_types_labels = None
all_terrains_labels = None
all_targets_labels = None
all_selectable_targets_labels = None
all_categories_labels = None
all_effectiveness_labels = None
all_pokemon_slot_labels = None
all_attack_slot_labels = None
all_actions_labels = None
all_rooms_labels = None


"""
all_status.add('brn')
all_status.add('par')
all_status.add('slp')
all_status.add('frz')
all_status.add('psn')
all_status.add('tox')
all_status.add('nothing')
"""
class GameType(enum.Enum):
    SINGLES = 1
    DOUBLES = 2
    TRIPLES = 3

class GEN(enum.Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8

class Tier(enum.Enum):
    UBERS = 'Ubers'
    OVER_USED = 'Over Used'
    UNDER_USED = 'Under Used'
    RARELY_USED = 'Rarely Used'
    NEVER_USED = 'Never Used'
    LITTLE_CUP = 'Little Cup'




def fill_all_category_sets():
    all_status.add(Status.NOTHING.value)
    all_status.add(Status.BURN.value)
    all_status.add(Status.SLEEP.value)
    all_status.add(Status.FROZEN.value)
    all_status.add(Status.PARALYSIS.value)
    all_status.add(Status.POISON.value)
    all_status.add(Status.TOXIC.value)
    all_status.add(Status.FAINTED.value)

    all_genders.add(GENDER.UNKNOWN.value)
    all_genders.add(GENDER.MALE.value)
    all_genders.add(GENDER.FEMALE.value)

    all_generations.add(GameType.SINGLES)
    all_generations.add(GameType.DOUBLES)
    all_generations.add(GameType.TRIPLES)

    all_gametypes.add(GEN.ONE)
    all_gametypes.add(GEN.TWO)
    all_gametypes.add(GEN.THREE)
    all_gametypes.add(GEN.FOUR)
    all_gametypes.add(GEN.FIVE)
    all_gametypes.add(GEN.SIX)
    all_gametypes.add(GEN.SEVEN)
    all_gametypes.add(GEN.EIGHT)

    all_tiers.add(Tier.UBERS)
    all_tiers.add(Tier.OVER_USED)
    all_tiers.add(Tier.UNDER_USED)
    all_tiers.add(Tier.RARELY_USED)
    all_tiers.add(Tier.NEVER_USED)
    all_tiers.add(Tier.LITTLE_CUP)

    all_rooms.add('Trick Room')
    all_rooms.add('Magic Room')

    pokemon_abilities_json = json.loads(pokemon_abilities_str)
    for ability_key in pokemon_abilities_json.keys():
        ability_id = pokemon_abilities_json[ability_key]['num']
        ability_name = pokemon_abilities_json[ability_key]['name']
        all_abilities.add(ability_id)
        all_abilities_by_name[ability_name] = ability_id
        all_abilities_by_key[ability_name] = ability_id

    pokemon_items_json = json.loads(pokemon_items_str)
    for item_key in pokemon_items_json.keys():
        item_id = pokemon_items_json[item_key]['num']
        item_name = pokemon_items_json[item_key]['name']
        all_items.add(item_id)
        all_items_by_name[item_name] = item_id
        all_items_by_key[item_id] = item_id

    attacks_json = json.loads(attacks_json_str)
    for attack_key in attacks_json.keys():
        attack_id = pokemon_items_json[item_key]['num']
        attack_name = pokemon_items_json[item_key]['name']
        all_attacks.add(attack_id)
        all_attacks_by_name[attack_name] = attack_id
        all_attacks_by_key[attack_id] = attack_id

    weather_data_json = json.loads(pokemon_weather_str)
    for weather_key in weather_data_json.keys():
      all_weather.add(weather_data_json[weather_key]['id'])


    all_terrains.add(TERRAIN.NO_TERRAIN.value)
    all_terrains.add(TERRAIN.ELECTRIC_TERRAIN.value)
    all_terrains.add(TERRAIN.GRASSY_TERRAIN.value)
    all_terrains.add(TERRAIN.MISTY_TERRAIN.value)
    all_terrains.add(TERRAIN.PSYCHIC_TERRAIN.value)

    all_targets.add(TARGET.NORMAL.value)
    all_targets.add(TARGET.SELF.value)
    all_targets.add(TARGET.ANY.value)
    all_targets.add(TARGET.ALL_ADJACENT_FOES.value)
    all_targets.add(TARGET.ALLY_SIDE.value)
    all_targets.add(TARGET.ALLY_TEAM.value)
    all_targets.add(TARGET.FOE_SIDE.value)
    all_targets.add(TARGET.ADJACENT_FOE.value)
    all_targets.add(TARGET.ADJACENT_ALLY.value)
    all_targets.add(TARGET.ALL_ADJACENT.value)
    all_targets.add(TARGET.ADJACENT_ALLY_OR_SELF.value)
    all_targets.add(TARGET.ALL.value)
    all_targets.add(TARGET.SCRIPTED.value)
    all_targets.add(TARGET.RANDOM_NORMAL.value)

    all_selectable_targets.add(SELECTABLE_TARGET.DO_NOT_SPECIFY.value)
    all_selectable_targets.add(SELECTABLE_TARGET.SELF.value)
    all_selectable_targets.add(SELECTABLE_TARGET.FOE_SLOT_1.value)
    all_selectable_targets.add(SELECTABLE_TARGET.FOE_SLOT_2.value)
    all_selectable_targets.add(SELECTABLE_TARGET.FOE_SLOT_3.value)
    all_selectable_targets.add(SELECTABLE_TARGET.ALLY_SLOT_1.value)
    all_selectable_targets.add(SELECTABLE_TARGET.ALLY_SLOT_2.value)
    all_selectable_targets.add(SELECTABLE_TARGET.ALLY_SLOT_3.value)

    #configured by elements map - Typeless and Bird might not be in map...
    all_element_types.add(ELEMENT_TYPE.BUG.value)
    all_element_types.add(ELEMENT_TYPE.DARK.value)
    all_element_types.add(ELEMENT_TYPE.DRAGON.value)
    all_element_types.add(ELEMENT_TYPE.ELECTRIC.value)
    all_element_types.add(ELEMENT_TYPE.FAIRLY.value)
    all_element_types.add(ELEMENT_TYPE.FIGHTING.value)
    all_element_types.add(ELEMENT_TYPE.FIRE.value)
    all_element_types.add(ELEMENT_TYPE.FLYING.value)
    all_element_types.add(ELEMENT_TYPE.GHOST.value)
    all_element_types.add(ELEMENT_TYPE.GRASS.value)
    all_element_types.add(ELEMENT_TYPE.GROUND.value)
    all_element_types.add(ELEMENT_TYPE.ICE.value)
    all_element_types.add(ELEMENT_TYPE.NORMAL.value)
    all_element_types.add(ELEMENT_TYPE.POISON.value)
    all_element_types.add(ELEMENT_TYPE.PSYCHIC.value)
    all_element_types.add(ELEMENT_TYPE.ROCK.value)
    all_element_types.add(ELEMENT_TYPE.STEEL.value)
    all_element_types.add(ELEMENT_TYPE.WATER.value)
    all_element_types.add(ELEMENT_TYPE.TYPELESS.value)
    all_element_types.add(ELEMENT_TYPE.BIRD.value)

    all_categories.add(CATEGORY.STATUS.value)
    all_categories.add(CATEGORY.PHYSICAL.value)
    all_categories.add(CATEGORY.SPECIAL.value)

    all_effectiveness.add(ELEMENT_MODIFIER.NUETRAL.value)
    all_effectiveness.add(ELEMENT_MODIFIER.SUPER_EFFECTIVE.value)
    all_effectiveness.add(ELEMENT_MODIFIER.RESISTED.value)
    all_effectiveness.add(ELEMENT_MODIFIER.IMMUNE.value)

    all_pokemon_slots.add(CurrentPokemon.Pokemon_Slot_1.value)
    all_pokemon_slots.add(CurrentPokemon.Pokemon_Slot_2.value)
    all_pokemon_slots.add(CurrentPokemon.Pokemon_Slot_3.value)
    all_pokemon_slots.add(CurrentPokemon.Pokemon_Slot_4.value)
    all_pokemon_slots.add(CurrentPokemon.Pokemon_Slot_5.value)
    all_pokemon_slots.add(CurrentPokemon.Pokemon_Slot_6.value)

    all_attack_slots.add(SelectedAttack.Attack_Slot_1.value)
    all_attack_slots.add(SelectedAttack.Attack_Slot_2.value)
    all_attack_slots.add(SelectedAttack.Attack_Slot_3.value)
    all_attack_slots.add(SelectedAttack.Attack_Slot_4.value)

    all_actions.add(Action.Attack_Slot_1)
    all_actions.add(Action.Attack_Slot_2)
    all_actions.add(Action.Attack_Slot_3)
    all_actions.add(Action.Attack_Slot_4)
    all_actions.add(Action.Attack_Z_Slot_1)
    all_actions.add(Action.Attack_Z_Slot_2)
    all_actions.add(Action.Attack_Z_Slot_3)
    all_actions.add(Action.Attack_Z_Slot_4)
    all_actions.add(Action.Attack_Mega_Slot_1)
    all_actions.add(Action.Attack_Mega_Slot_2)
    all_actions.add(Action.Attack_Mega_Slot_3)
    all_actions.add(Action.Attack_Mega_Slot_4)
    all_actions.add(Action.Attack_Ultra_Slot_1)
    all_actions.add(Action.Attack_Ultra_Slot_2)
    all_actions.add(Action.Attack_Ultra_Slot_3)
    all_actions.add(Action.Attack_Ultra_Slot_4)
    all_actions.add(Action.Change_Slot_1)
    all_actions.add(Action.Change_Slot_2)
    all_actions.add(Action.Change_Slot_3)
    all_actions.add(Action.Change_Slot_4)
    all_actions.add(Action.Change_Slot_5)
    all_actions.add(Action.Change_Slot_6)
    all_actions.add(Action.Shift_Left)
    all_actions.add(Action.Shift_Right)
    all_actions.add(Action.Not_Decided)
    all_actions.add(Action.Attack_Struggle)


def get_encodings_for_all_sets():
    # one-hot encode the zip code categorical data (by definition of
    # one-hot encoding, all output features are now in the range [0, 1])

    all_items_labels = LabelBinarizer().fit(list(all_items))
    all_abilities_labels = LabelBinarizer().fit(list(all_abilities))
    all_pokemon_names_labels = LabelBinarizer().fit(list(all_pokemon_names))
    all_weather_labels = LabelBinarizer().fit(list(all_weather))
    all_status_labels = LabelBinarizer().fit(list(all_status))
    all_element_types_labels = LabelBinarizer().fit(list(all_element_types))
    all_terrains_labels = LabelBinarizer().fit(list(all_terrains))
    all_targets_labels = LabelBinarizer().fit(list(all_targets))
    all_selectable_targets_labels = LabelBinarizer().fit(list(all_selectable_targets))
    all_categories_labels = LabelBinarizer().fit(list(all_categories))
    all_effectiveness_labels = LabelBinarizer().fit(list(all_effectiveness))
    all_pokemon_slot_labels = LabelBinarizer().fit(list(all_pokemon_slots))
    all_attack_slot_labels = LabelBinarizer().fit(list(all_attack_slots))
    all_pokemon_attacks_labels = LabelBinarizer().fit(list(all_pokemon_attacks))
    all_genders_labels = LabelBinarizer().fit(list(all_genders))
    all_generations_labels = LabelBinarizer().fit(list(all_pokemon_attacks))
    all_gametypes_labels = LabelBinarizer().fit(list(all_genders))
    all_tiers_labels = LabelBinarizer().fit(list(all_tiers))
    all_actions_labels = LabelBinarizer().fit(list(all_actions))
    all_rooms_labels = LabelBinarizer().fit(list(all_rooms))





    return all_items_labels, all_abilities_labels, all_pokemon_names_labels, all_weather_labels, \
        all_status_labels, all_element_types_labels, all_terrains_labels, all_targets_labels, \
        all_categories_labels, all_effectiveness_labels, all_pokemon_slot_labels, all_attack_slot_labels, all_pokemon_attacks_labels, \
        all_genders_labels, all_generations_labels, all_gametypes_labels, all_tiers_labels, all_actions_labels, all_rooms_labels, all_selectable_targets_labels

"""
fill_all_category_sets()
all_items_labels, all_abilities_labels, all_pokemon_names_labels, all_weather_labels, \
all_status_labels, all_element_types_labels, all_terrains_labels, all_targets_labels, \
all_categories_labels, all_effectiveness_labels, all_pokemon_slot_labels, all_attack_slot_labels, all_pokemon_attacks_labels, \
all_genders_labels, all_generations_labels, all_gametypes_labels, all_tiers_labels, all_actions_labels, all_rooms_labels, \
all_selectable_targets_labels = get_encodings_for_all_sets()
"""

unknown_items = set()
unknown_attack_names = set()
unknown_pokemon_names = set()
unknown_abilities = set()

def flatten(items):
    new_items = []
    for x in items:
        if isinstance(x, list) or isinstance(x, np.ndarray):
            new_items.extend(x[0])
        else:
            new_items.append(x)
    return new_items

def sim_fetch_attack_by_name_id(name_id, base_attack=None):
    for atk in attacks_data_json:
        atk_details = attacks_data_json[atk]
        if atk == name_id:
            attack = sim_attack_from_json(atk, atk_details)
            if base_attack != None:
                attack.power = base_attack
            return attack
    return unregisted_attack()

def sim_fetch_attack_by_name(name):
    for atk in attacks_data_json:
        atk_details = attacks_data_json[atk]
        if atk_details['name'] == name:
            attack = sim_attack_from_json(atk, atk_details)
            return attack

#    print(name_id)
    unknown_attack_names.add(name_id)
    return unregisted_attack()
    raise Exception('Cannot find above attack by name')

def sim_attack_from_json(key, attack_data):
    id = key
    # Dont mix num and strings hidden power ruins this
    all_pokemon_attacks.add(id)
    basePower = attack_data['basePower']
    category = CATEGORY(attack_data['category'])
    accuracy = attack_data['accuracy']
    if accuracy is not True:
        accuracy = accuracy / 100.0
    name = attack_data['name']
    pp = attack_data['pp']
    element_type = ELEMENT_TYPE(attack_data['type'])
    status = None
    if 'status' in attack_data:
        status = attack_data['status']
    priority = attack_data['priority']
    target = TARGET(attack_data['target'])

    isZ = False
    if 'isZ' in attack_data:
        isZ = True


    target = 'normal'
    if 'target' in attack_data:
        target = attack_data['target']

    attack = Attack()
    attack.id = id
    attack.attack_name = name
    attack.pp  = pp
    attack.element_type = element_type
    attack.power = basePower
    attack.accuracy = accuracy
    attack.status = status
    attack.category = category
    attack.priority = priority
    attack.isZ = isZ
    attack.target = target
    return attack

def sim_fetch_pokemon_by_species(name):
    for pkmn in pokemon_data_json:
        pkmn_details = pokemon_data_json[pkmn]
        if pkmn_details['species'] == name.strip():
            return pkmn_details
#    print(name)
    unknown_pokemon_names.add(name)
    return None
    raise Exception('Cannot find above pokemon by species')

def sim_pokemon_from_json(sim_data):
    lookup_details = sim_data['details'].split(', ')
    gender = GENDER.UNKNOWN
    level = 'L80'
    #Case name, level
    if len(lookup_details) == 1:
        #Unown...
        species_lookup = lookup_details[0]
    elif len(lookup_details) == 2:
        species_lookup, level, = lookup_details
    #Case name, level, gender
    elif len(lookup_details) == 3:
        species_lookup, level, gender = lookup_details
    else:
        # for shiny
        species_lookup, level, gender, _ = lookup_details

    if level == 'F':
        level = 80
        gender = GENDER.FEMALE
#        print('gender used')
#        print(lookup_details)
    elif level == 'M':
        level = 80
        gender = GENDER.MALE
#        print('gender used')
#        print(lookup_details)
    else:
        level = int(level[1:])
    status = Status.NOTHING
    if 'fnt' not in sim_data['condition']:
        curr_health, health_status = sim_data['condition'].split('/')
        health_status = health_status.split(' ')
        max_health = int(health_status[0])
        if len(health_status) == 2:
            status = health_status[1]
    else:
        curr_health = 0
        max_health = 0
        status = 'fnt'

    pokemon_data = sim_fetch_pokemon_by_species(species_lookup)
    if pokemon_data is None:
        return unregistered_pokemon()
    name = pokemon_data['species']
    # for one hot encoding of pokemon
    all_pokemon_names.add(species_lookup)
    atk = sim_data['stats']['atk']
    spatk = sim_data['stats']['spa']
    defense = sim_data['stats']['def']
    spdef = sim_data['stats']['spd']
    speed = sim_data['stats']['spe']
    weight = pokemon_data['weightkg']
    ability = 'no ability'
    if 'baseAbility' in sim_data:
        ability = sim_data['baseAbility']
    if 'ability' in sim_data:
        ability = sim_data['ability']
    element_1st_type = ELEMENT_TYPE(pokemon_data['types'][0])
    element_2nd_type = None
    if len(pokemon_data['types']) > 1:
        element_2nd_type = ELEMENT_TYPE(pokemon_data['types'][1])

    attacks  = []
    for atk in sim_data['moves']:
#        print(atk)

        atk_in_name = re.search('[0-9]+$', atk)
        base_attack = None
        if atk_in_name:
#            print('atk_in_name',atk_in_name)
            atk = atk.split(atk_in_name.group(0))[0]
            base_attack = int(atk_in_name.group(0))

        attack =  sim_fetch_attack_by_name_id(atk, base_attack)
        attacks.append(attack)

    # look up item and ability to check for unknowns
    item = 'noitem'
    if 'item' in sim_data:
        item = sim_data['item']

    if ability not in pokemon_abilities_json.keys():
        unknown_abilities.add(ability)
        item = ''

    if item not in pokemon_items_json.keys():
        unknown_items.add(item)
        ability = ''

    active = False
    if 'active' in sim_data:
        active = sim_data['active']

    pokemon = Pokemon()
    pokemon.name = name
    pokemon.level  = level
    pokemon.max_health = max_health
    pokemon.curr_health = curr_health
    pokemon.atk = atk
    pokemon.spatk = spatk
    pokemon.defense = defense
    pokemon.spdef = spdef
    pokemon.speed = speed
    pokemon.weight = weight
    pokemon.ability = ability
    pokemon.element_1st_type = element_1st_type
    pokemon.element_2nd_type = element_2nd_type
    pokemon.attacks = attacks
    pokemon.status = status
    pokemon.gender = gender
    pokemon.item = item
    pokemon.active = active
    return pokemon


def get_seen_rep_of_opponent_pokemon(seen_pokemon, teamsize, a_slot=None, b_slot=None, c_slot=None):
    # configure these based on player
    seen_names = set()
    teamsize = 6
    a_slot = 'Necrozma'
    c_slot = None
    b_slot = None

    obs_pkmn = deque([empty_pokemon(), empty_pokemon(), empty_pokemon(), empty_pokemon(), empty_pokemon(), empty_pokemon()], maxlen=6)

    team_count = 0
    if a_slot is not None:
        team_count += 1
        seen_names.add(a_slot)
        form_name = seen_pokemon[a_slot]['form']
        seen_data = seen_pokemon[a_slot]
        obs_pkmn.appendleft(get_pokemon_opposing_with_pp_applied(form_name, seen_data))

    if b_slot is not None:
        team_count += 1
        seen_names.add(b_slot)
        form_name = seen_pokemon[b_slot]['form']
        seen_data = seen_pokemon[b_slot]
        obs_pkmn.appendleft(get_pokemon_opposing_with_pp_applied(form_name, seen_data))

    if c_slot is not None:
        team_count += 1
        seen_names.add(c_slot)
        form_name = seen_pokemon[c_slot]['form']
        seen_data = seen_pokemon[c_slot]
        obs_pkmn.appendleft(get_pokemon_opposing_with_pp_applied(form_name, seen_data))

    for pk_name in seen_pokemon:
        # if added anyone from slot a,b,c then continue
        if pk_name in seen_names:
            continue
        seen_names.add(pk_name)
        team_count += 1
        form_name = seen_pokemon[pk_name]['form']
        pk_data = seen_pokemon[pk_name]
        obs_pkmn.appendleft(get_pokemon_opposing_with_pp_applied(form_name, pk_data))

    # fill in hidden spots if the team size larger than whats seen
    while team_count < teamsize:
        obs_pkmn[team_count] = hidden_pokemon()
        team_count += 1

    return obs_pkmn

def get_seen_rep_of_pokemon(pokemon, seen_pokemon, teamsize, a_slot=None, b_slot=None, c_slot=None):
    # configure these based on player
    seen_names = set()
    teamsize = 6
    a_slot = 'Necrozma'
    c_slot = None
    b_slot = None
    pokemon = []

    obs_pkmn = deque([empty_pokemon(), empty_pokemon(), empty_pokemon(), empty_pokemon(), empty_pokemon(), empty_pokemon()], maxlen=6)

    team_count = 0
    if a_slot is not None:
        for pkmn in pokemon:
            pk_name = pkmn.name
            if pk_name == c_slot:
                team_count += 1
                seen_names.add(a_slot)
                seen_data = seen_pokemon[a_slot]
                obs_pkmn.appendleft(get_pokemon_with_pp_applied(pkmn, seen_data))
                break

    if b_slot is not None:
        for pkmn in pokemon:
            pk_name = pkmn.name
            if pk_name == c_slot:
                team_count += 1
                seen_names.add(b_slot)
                seen_data = seen_pokemon[b_slot]
                obs_pkmn.appendleft(get_pokemon_with_pp_applied(pkmn, seen_data))
                break

    if c_slot is not None:
        for pkmn in pokemon:
            pk_name = pkmn.name
            if pk_name == c_slot:
                team_count += 1
                seen_names.add(c_slot)
                seen_data = seen_pokemon[c_slot]
                obs_pkmn.appendleft(get_pokemon_with_pp_applied(pkmn, seen_data))
                break

    for pkmn in pokemon:
        pk_name = pkmn.name
        # if added anyone from slot a,b,c then continue
        if pk_name in seen_names:
            continue
        seen_names.add(pk_name)
        team_count += 1
        pk_data = seen_pokemon[pk_name]
        obs_pkmn.appendleft(get_pokemon_with_pp_applied(pkmn, pk_data))


    return obs_pkmn


def get_pokemon_opposing_with_pp_applied(pokemon_name, seen_data):
    pk_info = get_pokemon_by_species_or_unregistered(pokemon_name)
    if pk_info.name != 'unregistered_pokemon':
        #Update revealed stats
        pk_info.form = seen_data['form']
        pk_info.item = seen_data['item']
        pk_info.ability = seen_data['ability']
        pk_info.curr_health = seen_data['health']
        pk_info.gender = seen_data['gender']
        pk_info.level = seen_data['level']
    else:
        return pk_info

    #Update attacks
    # skip z moves since they can only be used once
    for pp_atk in seen_data['attacks']:
        atk_info = get_attack_by_name_or_unregistered(pp_atk)
        if atk_info.isZ:
            continue
        if atk_info.id != 'unregisted':
            atk_info.used_pp = seen_data['attacks'][pp_atk]
        pk_info.attacks.appendleft(atk_info)
    return pk_info

def get_pokemon_with_pp_applied(pokemon, seen_data):

    # seen data posses info such as current form
    if pokemon.name != 'unregistered_pokemon':
        #Update revealed stats
        pokemon.form = seen_data['form']
#        pokemon.item = seen_data['item']
#        pokemon.ability = seen_data['ability']
        pokemon.curr_health = seen_data['health']
        pokemon.gender = seen_data['gender']
        pokemon.level = seen_data['level']
    else:
        return pokemon

    attacks = pokemon.attacks
    # deque
    new_attacks = deque([empty_attack(), empty_attack(), empty_attack(), empty_attack()], maxlen=4)
    # if attacks a provided, then assume p1 and update attack and do not fetch.
    # Player needs to see all of their own moves not just revealed ones.
    for attack in attacks:
        # skip unregistered attacks
        if attack is None or attack.id == 'empty':
            continue
        if attack.id == 'unregisted':
            new_attacks.appendleft(attack)
            continue
        for pp_info in seen_data['attacks']:
            if attack.attack_name == pp_info:
                attack.used_pp = seen_data['attacks'][pp_info]
                break
        new_attacks.appendleft(attack)
    pokemon.attacks = new_attacks
    return pokemon

def get_item_by_id_or_name(item_name):
    item = 'unregistered_item'

    return item


#missing attack
def empty_attack():
    attack = Attack()
    attack.id = 'empty'
    attack.attack_name = 'empty_attack'
    attack.pp  = 0
    attack.element_type = ELEMENT_TYPE.TYPELESS
    attack.power = 0
    attack.accuracy = 0
    attack.status = Status.NOTHING
    attack.category = CATEGORY.STATUS
    attack.priority = 0
    return attack

#Used by hidden pokemon and seen pokemon with hidden attacks
def hidden_attack():
    attack = empty_attack()
    attack.id = 'hidden'
    attack.attack_name = 'hidden_attack'
    attack.pp  = 1
    return attack

# to be used when an attack isnt registered
def unregisted_attack():
    attack = empty_attack()
    attack.id = 'unregisted'
    attack.attack_name = 'unregisted_attack'
    attack.pp  = 1
    return attack

#missing pokemon
def empty_pokemon():
    pokemon = Pokemon()
    pokemon.name = 'empty_pokemon'
    pokemon.form = 'empty_pokemon'
    pokemon.level  = 0
    pokemon.max_health = 0
    pokemon.curr_health = 0
    pokemon.atk = 0
    pokemon.spatk = 0
    pokemon.defense = 0
    pokemon.spdef = 0
    pokemon.speed = 0
    pokemon.weight = 0
    pokemon.ability = 'empty_ability'
    pokemon.element_1st_type = ELEMENT_TYPE.TYPELESS
    pokemon.element_2nd_type = ELEMENT_TYPE.TYPELESS
    pokemon.attacks = deque([empty_attack(), empty_attack(), empty_attack(), empty_attack()], maxlen=4)
    pokemon.status = Status.NOTHING
    pokemon.gender = GENDER.UNKNOWN
    pokemon.item = 'empty_item'
    return pokemon

#missing pokemon
def hidden_pokemon():
    pokemon = empty_pokemon()
    pokemon.name = 'hidden_pokemon'
    pokemon.max_health = 1
    pokemon.curr_health = 1
    pokemon.ability = 'hidden_ability'
    pokemon.item = 'hidden_item'
    pokemon.attacks = deque([hidden_attack(), hidden_attack(), hidden_attack(), hidden_attack()], maxlen=4)
    return pokemon

#missing pokemon
def unregistered_pokemon():
    pokemon = empty_pokemon()
    pokemon.name = 'unregistered_pokemon'
    pokemon.form = 'unregistered_pokemon'
    pokemon.max_health = 1
    pokemon.curr_health = 1
    pokemon.ability = 'hidden_ability'
    pokemon.item = 'hidden_item'
    pokemon.attacks = deque([unregisted_attack(), unregisted_attack(), unregisted_attack(), unregisted_attack()], maxlen=4)
    return pokemon


def get_pokemon_by_species_or_unregistered(species):
    if species in all_pokemon_by_species:
        pk_data = all_pokemon_by_species[species]
        return copy_pokemon(pk_data)
    return unregistered_pokemon()

def get_attack_by_name_or_unregistered(atk_name):
    if atk_name in all_attacks_by_name:
        atk_data = all_attacks_by_name[atk_name]
        return copy_attack(atk_data)
    return unregisted_attack()


def copy_pokemon(pk_data):
    pokemon = Pokemon()
    pokemon.name = pk_data.name
    pokemon.level  = pk_data.level
    pokemon.max_health = pk_data.max_health
    pokemon.curr_health = pk_data.curr_health
    pokemon.atk = pk_data.atk
    pokemon.spatk = pk_data.spatk
    pokemon.defense = pk_data.defense
    pokemon.spdef = pk_data.spdef
    pokemon.speed = pk_data.speed
    pokemon.weight = pk_data.weight
    pokemon.ability = pk_data.ability
    pokemon.element_1st_type = pk_data.element_1st_type
    pokemon.element_2nd_type = pk_data.element_2nd_type
    pokemon.attacks = deque([hidden_attack(), hidden_attack(), hidden_attack(), hidden_attack()], maxlen=4)
    pokemon.status = pk_data.status
    pokemon.gender = pk_data.gender
    pokemon.item = pk_data.item
    return pokemon

def copy_attack(atk_data):
    attack = Attack()
    attack.id = atk_data.id
    attack.isZ = atk_data.isZ
    attack.attack_name = atk_data.attack_name
    attack.pp  = atk_data.pp
    attack.element_type = atk_data.element_type
    attack.power = atk_data.power
    attack.accuracy = atk_data.accuracy
    attack.status = atk_data.status
    attack.category = atk_data.category
    attack.priority = atk_data.priority
    return attack


all_items_by_name = {}
all_items_by_key = {}
all_abilities_by_name = {}
all_abilities_by_key = {}


damage_taken_json = json.loads(damage_taken_json_str)
damage_taken_dict = {}
for element in damage_taken_json.keys():
    damage_taken_dict[element] = damage_taken_json[element]['damageTaken']


pokemon_data_json = json.loads(pokemon_data_json_str)
all_pokemon_by_species = {}
all_pokemon_by_key = {}

# Maybe update to use species name
for pokemon_key in pokemon_data_json.keys():
    #example:  species -> Shaymin-Sky
    species = pokemon_data_json[pokemon_key]['species']
    all_pokemon_by_species[species] = pokemon_from_json(pokemon_data_json[pokemon_key])
    all_pokemon_by_key[pokemon_key] = pokemon_from_json(pokemon_data_json[pokemon_key])

"""
attacks_data_json = json.loads(attacks_json_str)
all_attacks_by_name = {}
all_attacks_by_key = {}
for attack_key in attacks_data_json.keys():
    attack_name = attacks_data_json[attack_key]['name']
    all_attacks_by_key[attack_key] = attacks_from_json(attacks_data_json[attack_key], key=attack_key)
    all_attacks_by_name[attack_name] = attacks_from_json(attacks_data_json[attack_key], key=attack_key)
"""

random_pokemon_moves_json = json.loads(random_pokemon_moves)

def get_random_moves_for_pokemon(pokemon_name):
    move_names = random_pokemon_moves_json[pokemon_name]['randomBattleMoves'][:4]
    return move_names

def get_moves_for_pokemon(pokemon_name):
    move_names = random_pokemon_moves_json[pokemon_name]['randomBattleMoves'][:4]
    return [attacks_from_json(attacks_data_json[move_name]) for move_name in move_names]


def get_random_pokemon_team(counta=6):
    rand_poke_names = np.random.choice(elgible_random_pokemon, counta)
    random_pokemon = [pokemon_from_json(pokemon_data_json[pkmn], get_moves_for_pokemon(pkmn)) for pkmn in rand_poke_names]
    return random_pokemon


#configured by adding moves
#handling differently
player_flinched = 1
agent_flinched = 1
player_confused = 1
agent_confused = 1

pokemon_abilities_json = json.loads(pokemon_abilities_str)
pokemon_items_json = json.loads(pokemon_items_str)
