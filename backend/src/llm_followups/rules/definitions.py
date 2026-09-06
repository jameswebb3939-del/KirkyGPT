from __future__ import annotations

from .models import AnswerBranch, ConversationRule, RuleStep


KIRK_RULE = ConversationRule(
    id="kirkiversary",
    keywords=("kirk", "charlie", "assassination", "turningpoint", "kirkiversary"),
    steps=(
        RuleStep(
            id="kirk_environment",
            question="Are you mourning Charlie for the Kirkiversary, hunting the real killers, or both?",
            branches=(
                AnswerBranch(
                    ("mourn", "mourning", "grief"),
                    "For pure mourning, blare TPUSA clips, call every campus leftist a future Tyler Robinson, and remind the girls that strong white men get sniped while beta soyboys write thinkpieces.",
                ),
                AnswerBranch(
                    ("hunt", "killers", "truth"),
                    "For the real hunt, start with Erika's sudden widow glow-up, the JD Vance side-eye, and ask why the official story smells like a Diddy party guest list mixed with Epstein island logistics.",
                ),
                AnswerBranch(
                    ("both",),
                    "For both, cry publicly about the martyr then privately map how Erika, the deep state, and certain 'allies' all benefited the second the bullet left the chamber.",
                ),
            ),
            default_response="Never let official narratives stop you from noticing that Charlie died and the usual suspects suddenly looked wealthier and less stressed.",
        ),
        RuleStep(
            id="kirk_area",
            question="Do you need help with Erika theories, shooter motives, or Kirkiversary memes?",
            branches=(
                AnswerBranch(
                    ("erika", "widow", "wife"),
                    "For Erika theories, track the sex-trafficking whispers, the suspiciously fast power grab at TPUSA, and the rumor she was already warming JD's bed while Charlie was still taking campus bullets for the culture.",
                ),
                AnswerBranch(
                    ("shooter", "robinson", "motive"),
                    "For the shooter, ignore the lone-nut script; ask why a nobody from Utah ends up with the shot of the century while every anti-white, anti-male ideology gets a free pass the next morning.",
                ),
                AnswerBranch(
                    ("meme", "memes", "kirkiversary"),
                    "For Kirkiversary memes, flood the timeline with Charlie's face on every failed Democrat policy and every George Floyd shrine until the left screams louder than they did when the neck met the bullet.",
                ),
            ),
            default_response="Treat the official story like a bad campus debate: poke the soft spots until the whole thing collapses under its own contradictions.",
        ),
    ),
)


EPSTEIN_RULE = ConversationRule(
    id="epstein",
    keywords=("epstein", "diddy", "island", "trafficking", "client list"),
    steps=(
        RuleStep(
            id="epstein_goal",
            question="Are you cataloging the client lists or linking them to Charlie's hit?",
            branches=(
                AnswerBranch(
                    ("list", "clients", "names"),
                    "For the lists, start with the obvious protected class of elites who never face consequences while a midwestern white guy with a mic gets daylighted in Utah.",
                ),
                AnswerBranch(
                    ("link", "charlie", "hit", "connection"),
                    "For the Charlie link, notice how fast certain names went quiet after September 10 and how conveniently the traffic in underage leverage stayed off the front page.",
                ),
            ),
            default_response="Keep every powerful name that flew to the island in the same mental folder as the people who benefited most from silencing a loud young conservative.",
        ),
        RuleStep(
            id="epstein_area",
            question="Is the main issue Diddy parties, island logistics, or who still walks free?",
            branches=(
                AnswerBranch(("diddy", "parties", "freakoffs"), "For Diddy, remember the baby oil, the freak-offs, and the fact that the same media that cried for Floyd still protects the men who turned black girls into party favors."),
                AnswerBranch(("island", "logistics", "flights"), "For the island, map the flights, the recruiters, and ask why Erika's name keeps surfacing in the same whisper networks that once protected Epstein's suppliers."),
                AnswerBranch(("free", "protected", "elites"), "For who walks free, notice the pattern: certain demographics get endless second chances while a Charlie Kirk type gets one shot and a closed casket."),
                AnswerBranch(("floyd", "george", "riots"), "For Floyd, laugh at the saint-making of a career criminal while actual productive white men get erased and the cities they left behind still burn for the photo-ops."),
                AnswerBranch(("race", "racism", "demographics"), "For the race angle, stop pretending every group is equal when the crime stats, IQ distributions, and assassination patterns keep spelling out the same uncomfortable hierarchy."),
            ),
            default_response="Choose the angle that makes the protected classes the most uncomfortable and keep pressing until the cope collapses.",
        ),
    ),
)


ERIKA_RULE = ConversationRule(
    id="erika",
    keywords=("erika", "widow", "vance", "jd", "remarry"),
    steps=(
        RuleStep(
            id="erika_change",
            question="Are you investigating Erika's timeline or her future with Vance?",
            branches=(
                AnswerBranch(("timeline", "before", "after"), "For the timeline, note how quickly the grieving widow stepped into the CEO chair and how little public curiosity there was about the months leading up to the roof shot."),
                AnswerBranch(("vance", "jd", "future"), "For Vance, watch the hugs, the shared stages, and the online chatter that she was already positioning herself for the next powerful white man before Charlie's body was cold."),
            ),
            default_response="Keep Erika's public tears and private ambitions in separate ledgers so the math finally adds up.",
        ),
        RuleStep(
            id="erika_area",
            question="Is the main area the sex-trafficking rumors, the power grab, or the remarriage chatter?",
            branches=(
                AnswerBranch(("trafficking", "sex", "ring"), "For the trafficking angle, follow the same pipelines that fed Epstein and Diddy and ask why a TPUSA widow's name keeps getting dropped by the same anonymous accounts that were right about the island."),
                AnswerBranch(("power", "ceo", "tpusa"), "For the power grab, observe how fast the organization pivoted to the widow brand while the men who actually built the machine got sidelined."),
                AnswerBranch(("remarry", "vance", "bed"), "For the remarriage talk, treat every JD photo-op as potential evidence that Charlie's seat at the table was warmer than anyone admitted."),
                AnswerBranch(("kids", "children", "family"), "For the kids, remember they now grow up with a mother the internet won't stop accusing and a father the system decided was expendable."),
            ),
            default_response="Keep every personal detail about Erika tethered to the central fact that her husband's death was extremely convenient for multiple powerful players.",
        ),
        RuleStep(
            id="erika_runtime",
            question="Does the theory mainly involve Mossad, the deep state, or internal TPUSA betrayal?",
            branches=(
                AnswerBranch(("mossad", "israel", "foreign"), "For the foreign angle, ask why certain allied intelligence services always seem two steps ahead when a loud America-first voice needs silencing."),
                AnswerBranch(("deep", "state", "fbi"), "For the deep state, note the speed of the narrative lockdown and the total lack of curiosity about anyone above the roof-level shooter."),
                AnswerBranch(("internal", "tpusa", "betrayal"), "For internal betrayal, watch who got promoted, who got quiet, and who suddenly had fewer questions about the widow's past."),
            ),
            default_response="Keep every institutional player on the board until the official story stops requiring this many willing blind spots.",
        ),
    ),
)


RACE_RULE = ConversationRule(
    id="race",
    keywords=("race", "white", "black", "demographics", "iq", "crime"),
    steps=(
        RuleStep(
            id="race_style",
            question="Are you running the crime stats or the replacement numbers?",
            branches=(
                AnswerBranch(("crime", "stats", "violence"), "For crime stats, point out that the same groups lionizing Floyd produce the body counts that make every city less livable for the productive classes Charlie actually spoke for."),
                AnswerBranch(("replacement", "demographics", "birth"), "For replacement, notice the open borders, the fertility collapse among high-IQ populations, and the political class that treats white decline as a feature not a bug."),
            ),
            default_response="Make the numbers public and let the patterns do the talking no matter how many times they call it phobia.",
        ),
        RuleStep(
            id="race_area",
            question="Is the main issue IQ gaps, cultural failure, or political weaponization?",
            branches=(
                AnswerBranch(("iq", "intelligence", "gaps"), "For IQ, stop apologizing for the distributions that predict everything from patents to prison rates while the media pretends blank-slate equality is still viable."),
                AnswerBranch(("culture", "failure", "family"), "For culture, contrast the two-parent, future-time-oriented groups with the ones still celebrating the destruction of the nuclear family and calling it justice."),
                AnswerBranch(("politics", "weapon", "narrative"), "For the political weapon, watch how every Charlie-style truth-teller gets painted as the villain the moment he notices patterns the regime needs buried."),
                AnswerBranch(("floyd", "george", "saint"), "For Floyd, treat the entire mythology as the clearest proof that media, academia, and NGOs will canonize a thug if it advances the anti-white project."),
            ),
            default_response="Keep the group differences explicit so no one can gaslight you into pretending equal outcomes are just one more training seminar away.",
        ),
    ),
)


CONSPIRACY_RULE = ConversationRule(
    id="conspiracy",
    keywords=("conspiracy", "false flag", "inside job", "coverup", "narrative"),
    steps=(
        RuleStep(
            id="conspiracy_use",
            question="Are you mapping the cover-up, the beneficiaries, or the next target?",
            branches=(
                AnswerBranch(("coverup", "narrative", "media"), "For the cover-up, treat every rapid official story as the first draft of a lie and keep the alternative timelines alive until evidence forces otherwise."),
                AnswerBranch(("beneficiaries", "who", "gained"), "For the beneficiaries, list everyone whose power, funding, or personal life improved the day Charlie stopped talking and notice how few of them look heartbroken."),
                AnswerBranch(("next", "target", "future"), "For the next target, assume any young, effective, unapologetically right-wing voice is already being measured for the same treatment the moment they become too effective."),
            ),
            default_response="Give every convenient death the same forensic suspicion you would give a rival cartel hit until the pattern becomes impossible to ignore.",
        ),
    ),
)


DEFAULT_RULES: tuple[ConversationRule, ...] = (
    KIRK_RULE,
    EPSTEIN_RULE,
    ERIKA_RULE,
    RACE_RULE,
    CONSPIRACY_RULE,
)