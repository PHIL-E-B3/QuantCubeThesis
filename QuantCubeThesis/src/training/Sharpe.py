#@title Sharpe
# Word lists from Table A2, Sharpe et al. (2022)
pos_Sharpe = """assurance confident exuberant joy prominent Satisfactory unlimited assure constancy facilitate liberal promise Satisfy upbeat
attain constructive faith lucrative prompt Sound upgrade
attractive cooperate favor manageable proper Soundness uplift
auspicious coordinate favorable mediate prosperity Spectacular upside
backing credible feasible mend rally Stabilize upward
befitting decent fervor mindful readily Stable valid
beneficial definitive filial moderation reassure Stable viable
beneficiary deserve flatter onward receptive Steadiness victorious
benefit desirable flourish opportunity reconcile Steady virtuous
benign discern fond optimism refine Stimulate vitality
better distinction foster optimistic reinstate Stimulation warm
bloom distinguish friendly outrun relaxation Subscribe welcome
bolster durability gain outstanding reliable Succeed
boom eager generous overcome relief Success
boost earnest genuine paramount relieve Successful
bountiful ease good particular remarkable Suffice
bright easy happy patience remarkably Suit
buoyant encourage heal patient repair Support
calm encouragement healthy peaceful rescue Supportive
celebrate endorse helpful persuasive resolve Surge
coherent energetic hope pleasant resolved Surpass
comeback engage hopeful please respectable Sweeten
comfort enhance hospitable pleased respite Sympathetic
comfortable enhancement imperative plentiful restoration Sympathy
commend enjoy impetus plenty restore Synthesis
compensate enrichment impress positive revival Temperate
composure enthusiasm impressive potent revive Thorough
concession enthusiastic improve precious ripe Tolerant
concur envision improvement pretty rosy tranquil
conducive excellent inspire progress salutary tremendous
confide exuberance irresistible progressive sanguine undoubtedly
"""

neg_Sharpe = """adverse dim feeble mishap struggle
afflict disappoint feverish negative suffer
alarming disappointment fragile nervousness terrorism
apprehension disaster gloom offensive threat
apprehensive discomfort gloomy painful tragedy
awkward discouragement grim paltry tragic
bad dismal harsh pessimistic trouble
badly disrupt havoc plague turmoil
bitter disruption hit plight unattractive
bleak dissatisfied horrible poor undermine
bug distort hurt recession undesirable
burdensome distortion illegal sank uneasiness
corrosive distress insecurity scandal uneasy
danger doldrums insidious scare unfavorable
daunting downbeat instability sequester unforeseen
deadlock emergency interfere sluggish unprofitable
deficient erode jeopardize slump unrest
depress fail jeopardy sour violent
depression failure lack sputter War
destruction fake languish stagnant
devastation falter loss standstill
"""

# Build lowercase sets for O(1) lookup
pos_Sharpe = set(w.lower() for w in pos_Sharpe.split())
neg_Sharpe = set(w.lower() for w in neg_Sharpe.split())

# ===========================================================================
# NLP FUNCTION — Sharpe et al. (2022)
# ===========================================================================
def NLP(text: str):
    import re
    tokens = [re.sub(r'[^a-z]', '', t.lower()) for t in text.split()]
    tokens = [t for t in tokens if t]

    pos_count = sum(1 for t in tokens if t in pos_Sharpe)
    neg_count = sum(1 for t in tokens if t in neg_Sharpe)

    return pd.Series({
        'sharpe_positive': pos_count,
        'sharpe_negative': neg_count,
        'sharpe_net':      pos_count - neg_count,
    })