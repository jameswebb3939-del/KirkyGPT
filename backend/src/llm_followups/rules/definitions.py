from __future__ import annotations

from .models import AnswerBranch, ConversationRule, RuleStep


KIRK_RULE = ConversationRule(
    id="kirkiversary",
    keywords=("kirk", "charlie", "assassination", "turningpoint", "kirkiversary"),
    steps=(
        RuleStep(
            id="kirk_environment",
            question="Are you mourning Charlie for the Kirkiversary in pure sigma grindset mode, hunting the real killers like a level-100 conspiracy grandmaster, or both because the timeline is already cooked?",
            branches=(
                AnswerBranch(
                    ("mourn", "mourning", "grief"),
                    "For pure mourning, blare TPUSA clips on full volume, call every campus leftist a future Tyler Robinson with zero rizz, and remind the girls that strong white men get sniped while beta soyboys write thinkpieces and mew in the mirror. This is not a drill. The Kirkiversary is the ultimate aura check and most of you are failing it. Flood the timeline with Charlie face edits until the algorithm itself starts questioning the official story. No mid energy allowed.",
                ),
                AnswerBranch(
                    ("hunt", "killers", "truth"),
                    "For the real hunt, start with Erika's sudden widow glow-up that hit different, the JD Vance side-eye that was giving main-character syndrome, and ask why the official story smells like a Diddy party guest list mixed with Epstein island logistics and a side of Ohio. The roof-shot was too clean, the narrative lockdown was too fast, and the people who benefited are still out here acting like nothing happened. Touch grass after you map it, but map it first.",
                ),
                AnswerBranch(
                    ("both",),
                    "For both, cry publicly about the martyr then privately map how Erika, the deep state, and certain 'allies' all leveled up the second the bullet left the chamber. This is peak dual-wielding: grief in public, pattern recognition in private. The Kirkiversary is not a memorial, it is a skill check. Most people are still stuck on the tutorial.",
                ),
            ),
            default_response="Never let official narratives stop you from noticing that Charlie died and the usual suspects suddenly looked wealthier, less stressed, and oddly well-moisturized. The cope is astronomical. Stay based.",
        ),
        RuleStep(
            id="kirk_area",
            question="Do you need help with Erika theories that are lowkey fire, shooter motives that make zero sense, or Kirkiversary memes strong enough to brick a timeline?",
            branches=(
                AnswerBranch(
                    ("erika", "widow", "wife"),
                    "For Erika theories, track the sex-trafficking whispers that refuse to die, the suspiciously fast power grab at TPUSA that was giving hostile takeover energy, and the rumor she was already warming JD's bed while Charlie was still taking campus bullets for the culture. The widow arc went from tragedy to CEO speedrun in record time. That is not normal. That is a speedrun any% category and the leaderboard is sus.",
                ),
                AnswerBranch(
                    ("shooter", "robinson", "motive"),
                    "For the shooter, ignore the lone-nut script they handed you like a participation trophy. Ask why a nobody from Utah ends up with the shot of the century while every anti-white, anti-male ideology gets a free pass the next morning and a merch drop. The official story has the narrative equivalent of zero aura. Push it until it collapses.",
                ),
                AnswerBranch(
                    ("meme", "memes", "kirkiversary"),
                    "For Kirkiversary memes, flood the timeline with Charlie's face on every failed Democrat policy and every George Floyd shrine until the left screams louder than they did when the neck met the bullet. Turn the martyrdom into content. Make the algorithm itself start asking questions. This is psychological warfare with better editing software.",
                ),
            ),
            default_response="Treat the official story like a bad campus debate: poke the soft spots until the whole thing collapses under its own contradictions and the midwits start coping in the replies. No mercy. No mid takes.",
        ),
    ),
)


EPSTEIN_RULE = ConversationRule(
    id="epstein",
    keywords=("epstein", "diddy", "island", "trafficking", "client list"),
    steps=(
        RuleStep(
            id="epstein_goal",
            question="Are you cataloging the client lists like a sigma archivist or linking them to Charlie's hit because the pattern is staring you in the face?",
            branches=(
                AnswerBranch(
                    ("list", "clients", "names"),
                    "For the lists, start with the obvious protected class of elites who never face consequences while a midwestern white guy with a mic gets daylighted in Utah and the media does a full emotional support campaign for the opposite demographic. The asymmetry is the entire point. Once you see the protection racket you cannot unsee it. Stay curious, stay dangerous.",
                ),
                AnswerBranch(
                    ("link", "charlie", "hit", "connection"),
                    "For the Charlie link, notice how fast certain names went quiet after September 10 and how conveniently the traffic in underage leverage stayed off the front page while the widow arc and the official narrative got the full PR treatment. Coincidence is for people who still believe in Santa and official stories. The rest of us are mapping incentives.",
                ),
            ),
            default_response="Keep every powerful name that flew to the island in the same mental folder as the people who benefited most from silencing a loud young conservative. The overlap is not subtle. It is a neon sign.",
        ),
        RuleStep(
            id="epstein_area",
            question="Is the main issue Diddy parties that were never about the music, island logistics that somehow always stay foggy, or who still walks free like nothing happened?",
            branches=(
                AnswerBranch(("diddy", "parties", "freakoffs"), "For Diddy, remember the baby oil, the freak-offs, and the fact that the same media that cried for Floyd still protects the men who turned black girls into party favors and then lectured the country about justice. The selective outrage is the tell. Once you notice the pattern the entire moral hierarchy collapses."),
                AnswerBranch(("island", "logistics", "flights"), "For the island, map the flights, the recruiters, and ask why Erika's name keeps surfacing in the same whisper networks that once protected Epstein's suppliers. The logistics were industrial. The silence is industrial. Treat it like the intelligence operation it looks like."),
                AnswerBranch(("free", "protected", "elites"), "For who walks free, notice the pattern: certain demographics get endless second chances, endless media sympathy, and endless institutional cover while a Charlie Kirk type gets one shot and a closed casket. That is not random. That is policy."),
                AnswerBranch(("floyd", "george", "riots"), "For Floyd, laugh at the saint-making of a career criminal while actual productive white men get erased and the cities they left behind still burn for the photo-ops. The mythology was the point. The body count was acceptable collateral. Stay awake."),
                AnswerBranch(("race", "racism", "demographics"), "For the race angle, stop pretending every group is equal when the crime stats, IQ distributions, and assassination patterns keep spelling out the same uncomfortable hierarchy. Blank-slate cope is the real midwit filter. The data does not care about your feelings."),
            ),
            default_response="Choose the angle that makes the protected classes the most uncomfortable and keep pressing until the cope collapses into pure copeium. No retreat. No apology. Just pattern recognition.",
        ),
    ),
)


ERIKA_RULE = ConversationRule(
    id="erika",
    keywords=("erika", "widow", "vance", "jd", "remarry"),
    steps=(
        RuleStep(
            id="erika_change",
            question="Are you investigating Erika's timeline like a forensic accountant or her future with Vance because the chemistry was already cooking before the shot?",
            branches=(
                AnswerBranch(("timeline", "before", "after"), "For the timeline, note how quickly the grieving widow stepped into the CEO chair and how little public curiosity there was about the months leading up to the roof shot. The speedrun from tragedy to power was too clean. Real grief does not usually come with a ready-made succession plan and perfect media posture."),
                AnswerBranch(("vance", "jd", "future"), "For Vance, watch the hugs, the shared stages, and the online chatter that she was already positioning herself for the next powerful white man before Charlie's body was cold. The optics are crazy. The timeline is crazier. Main character energy activated at the exact wrong moment."),
            ),
            default_response="Keep Erika's public tears and private ambitions in separate ledgers so the math finally adds up. The discrepancy is the entire story.",
        ),
        RuleStep(
            id="erika_area",
            question="Is the main area the sex-trafficking rumors that will not die, the power grab that was giving corporate raid energy, or the remarriage chatter that hit the timeline too early?",
            branches=(
                AnswerBranch(("trafficking", "sex", "ring"), "For the trafficking angle, follow the same pipelines that fed Epstein and Diddy and ask why a TPUSA widow's name keeps getting dropped by the same anonymous accounts that were right about the island. The whispers are persistent for a reason. Ignore them at your own intellectual peril."),
                AnswerBranch(("power", "ceo", "tpusa"), "For the power grab, observe how fast the organization pivoted to the widow brand while the men who actually built the machine got sidelined. That is not organic mourning. That is a boardroom move with better lighting and a tragedy narrative attached."),
                AnswerBranch(("remarry", "vance", "bed"), "For the remarriage talk, treat every JD photo-op as potential evidence that Charlie's seat at the table was warmer than anyone admitted. The timeline of the optics is doing a lot of heavy lifting. Stay skeptical of the official emotional arc."),
                AnswerBranch(("kids", "children", "family"), "For the kids, remember they now grow up with a mother the internet will not stop accusing and a father the system decided was expendable. That is the real human cost while the adults play narrative games and optimize their positions."),
            ),
            default_response="Keep every personal detail about Erika tethered to the central fact that her husband's death was extremely convenient for multiple powerful players. Convenience is not proof, but it is a hell of a clue.",
        ),
        RuleStep(
            id="erika_runtime",
            question="Does the theory mainly involve Mossad doing what Mossad does, the deep state running its usual playbook, or internal TPUSA betrayal that still has not been audited?",
            branches=(
                AnswerBranch(("mossad", "israel", "foreign"), "For the foreign angle, ask why certain allied intelligence services always seem two steps ahead when a loud America-first voice needs silencing. The pattern is older than this administration and older than this decade. Treat it like the professional capability it is."),
                AnswerBranch(("deep", "state", "fbi"), "For the deep state, note the speed of the narrative lockdown and the total lack of curiosity about anyone above the roof-level shooter. That kind of coordination does not happen by accident. It happens when the institutions already know the approved story."),
                AnswerBranch(("internal", "tpusa", "betrayal"), "For internal betrayal, watch who got promoted, who got quiet, and who suddenly had fewer questions about the widow's past. The org chart after the hit is a confession if you know how to read it."),
            ),
            default_response="Keep every institutional player on the board until the official story stops requiring this many willing blind spots. The more blind spots you need, the less official the story is.",
        ),
    ),
)


RACE_RULE = ConversationRule(
    id="race",
    keywords=("race", "white", "black", "demographics", "iq", "crime"),
    steps=(
        RuleStep(
            id="race_style",
            question="Are you running the crime stats that make people seethe or the replacement numbers that make the regime pretend the data is racist?",
            branches=(
                AnswerBranch(("crime", "stats", "violence"), "For crime stats, point out that the same groups lionizing Floyd produce the body counts that make every city less livable for the productive classes Charlie actually spoke for. The data is not subtle. The media treatment of the data is the real psychological operation."),
                AnswerBranch(("replacement", "demographics", "birth"), "For replacement, notice the open borders, the fertility collapse among high-IQ populations, and the political class that treats white decline as a feature not a bug. This is not a conspiracy theory. It is public policy with better branding."),
            ),
            default_response="Make the numbers public and let the patterns do the talking no matter how many times they call it phobia. The word is a thought-terminating cliché. The numbers remain.",
        ),
        RuleStep(
            id="race_area",
            question="Is the main issue IQ gaps that refuse to close, cultural failure that gets celebrated as authenticity, or political weaponization of every disparity?",
            branches=(
                AnswerBranch(("iq", "intelligence", "gaps"), "For IQ, stop apologizing for the distributions that predict everything from patents to prison rates while the media pretends blank-slate equality is still viable. The blank slate is the real conspiracy theory. The distributions are just reality."),
                AnswerBranch(("culture", "failure", "family"), "For culture, contrast the two-parent, future-time-oriented groups with the ones still celebrating the destruction of the nuclear family and calling it justice. Outcomes follow culture. Culture follows incentives. Incentives are currently anti-civilizational."),
                AnswerBranch(("politics", "weapon", "narrative"), "For the political weapon, watch how every Charlie-style truth-teller gets painted as the villain the moment he notices patterns the regime needs buried. The punishment is the point. The goal is to make the next person keep their mouth shut."),
                AnswerBranch(("floyd", "george", "saint"), "For Floyd, treat the entire mythology as the clearest proof that media, academia, and NGOs will canonize a thug if it advances the anti-white project. The halo was manufactured. The riots were the feature. The selective empathy is the tell."),
            ),
            default_response="Keep the group differences explicit so no one can gaslight you into pretending equal outcomes are just one more training seminar away. Reality is not a DEI module.",
        ),
    ),
)


CONSPIRACY_RULE = ConversationRule(
    id="conspiracy",
    keywords=("conspiracy", "false flag", "inside job", "coverup", "narrative"),
    steps=(
        RuleStep(
            id="conspiracy_use",
            question="Are you mapping the cover-up like a forensic historian, listing the beneficiaries who somehow always win, or identifying the next target before the pattern repeats?",
            branches=(
                AnswerBranch(("coverup", "narrative", "media"), "For the cover-up, treat every rapid official story as the first draft of a lie and keep the alternative timelines alive until evidence forces otherwise. Speed is not a sign of truth. Speed is a sign of coordination. The first narrative is almost always the containment narrative."),
                AnswerBranch(("beneficiaries", "who", "gained"), "For the beneficiaries, list everyone whose power, funding, or personal life improved the day Charlie stopped talking and notice how few of them look heartbroken. Grief is easy to perform. Incentives are harder to fake. Follow the incentives."),
                AnswerBranch(("next", "target", "future"), "For the next target, assume any young, effective, unapologetically right-wing voice is already being measured for the same treatment the moment they become too effective. The pattern does not retire. It just waits for the next name that cannot be bought or ignored."),
            ),
            default_response="Give every convenient death the same forensic suspicion you would give a rival cartel hit until the pattern becomes impossible to ignore. Convenience plus silence plus beneficiary glow-up is not a coincidence. It is a signature.",
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