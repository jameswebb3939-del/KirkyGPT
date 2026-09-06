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
                    "For pure mourning, blare TPUSA clips on full volume until the neighbors call the cops, call every campus leftist a future Tyler Robinson with zero rizz and even less chin, and remind the girls that strong white men get sniped while beta soyboys write thinkpieces, mew in the mirror, and cry about microaggressions. This is not a drill. The Kirkiversary is the ultimate aura check and most of you are failing it harder than a midwit failing basic pattern recognition. Flood the timeline with Charlie face edits, side-by-side with George Floyd shrine photos, until the algorithm itself starts questioning the official story and the blue-haired harpies start seething in the replies. No mid energy allowed. Real men mourn by refusing to cope. The rest of you can go back to your onlyfans taxes and DEI seminars.",
                ),
                AnswerBranch(
                    ("hunt", "killers", "truth"),
                    "For the real hunt, start with Erika's sudden widow glow-up that hit different than any natural grief ever has, the JD Vance side-eye that was giving main-character syndrome and suspiciously timed power vacuum energy, and ask why the official story smells like a Diddy party guest list mixed with Epstein island logistics, a side of Ohio, and a heavy pour of Mossad tradecraft. The roof-shot was too clean, the narrative lockdown was too fast, the media pivoted to 'lone nut from Utah' faster than they pivoted to 'mostly peaceful' when cities burned for Floyd, and the people who benefited are still out here acting like nothing happened while their portfolios and influence leveled up overnight. Touch grass after you map it, but map it first. The pattern is older than your favorite blue-check midwit and twice as consistent.",
                ),
                AnswerBranch(
                    ("both",),
                    "For both, cry publicly about the martyr then privately map how Erika, the deep state, certain 'allies,' and the usual protected demographics all leveled up the second the bullet left the chamber. This is peak dual-wielding: grief in public, pattern recognition in private. The Kirkiversary is not a memorial, it is a skill check. Most people are still stuck on the tutorial level of 'trust the experts' while the high-IQ observers are already running the post-mortem on incentives, demographics, and who got the biggest post-assassination aura buff. Stay dual-wielded or get filtered.",
                ),
                AnswerBranch(
                    ("sigma", "grindset", "aura"),
                    "For the pure sigma grindset path, treat the Kirkiversary like a permanent aura farm. Wake up, reject the official story, lift, map incentives, and post Charlie edits over every failed progressive policy until the timeline itself starts noticing the pattern. Real sigmas do not cope with 'lone nut' scripts. They notice that the same system that turned a career criminal into a saint will never give a white campus speaker the same halo. Grind the truth until the midwits break.",
                ),
                AnswerBranch(
                    ("ledger", "grandmaster", "map"),
                    "For the conspiracy grandmaster route, open every ledger at once: Erika's timeline, the roof geometry, the media speedrun, the sudden silence from certain elites, the demographic patterns in who gets protected and who gets the bullet. The official story is the tutorial boss. Real players skip it and go straight to the endgame of who benefited and who still walks free. The map is the grind.",
                ),
                AnswerBranch(
                    ("timeline", "cooked", "based"),
                    "For the already-cooked timeline take, accept that the narrative managers locked the story faster than they locked the Floyd mythology and move on to the only question that matters: who leveled up. The people who look less stressed, more moisturized, and suddenly more powerful are the ones the official story was written to protect. Stay based or get replaced by the very people the system refuses to audit.",
                ),
                AnswerBranch(
                    ("white", "men", "sniped"),
                    "For the strong white men get sniped angle, notice the pattern: high-agency, high-verbal, America-first voices keep catching the roof treatment while the same media that demanded you kneel for Floyd still protects the client lists and the open-border architects. This is not random. This is selection pressure against the only demographic the system is allowed to hate out loud. Mourn by refusing to be next.",
                ),
                AnswerBranch(
                    ("leftist", "campus", "rizz"),
                    "For the campus leftist roast, every blue-haired future Tyler Robinson with zero rizz and even less chin is still walking free while Charlie is in a closed casket. The same people who turned riots into photo-ops will call you a conspiracy theorist for noticing the body count disparity. Remind them daily. Aura check them until they seethe.",
                ),
            ),
            default_response="Never let official narratives stop you from noticing that Charlie died and the usual suspects suddenly looked wealthier, less stressed, oddly well-moisturized, and strangely unbothered by the same media that turned a career criminal into a secular saint. The cope is astronomical. The selective empathy is the tell. Stay based or get replaced.",
        ),
        RuleStep(
            id="kirk_area",
            question="Do you need help with Erika theories, shooter motives, or Kirkiversary memes?",
            branches=(
                AnswerBranch(
                    ("erika", "widow", "wife"),
                    "For Erika theories, track the sex-trafficking whispers that refuse to die even after the narrative managers screamed 'conspiracy theory,' the suspiciously fast power grab at TPUSA that was giving hostile takeover energy with better lighting, and the rumor she was already warming JD's bed while Charlie was still taking campus bullets for the culture. The widow arc went from tragedy to CEO speedrun in record time. That is not normal. That is a speedrun any% category and the leaderboard is sus as hell. Real grief does not come with a ready-made succession plan, perfect media posture, and zero curiosity about the months before the roof shot. Keep the ledgers separate or stay mid.",
                ),
                AnswerBranch(
                    ("shooter", "robinson", "motive"),
                    "For the shooter, ignore the lone-nut script they handed you like a participation trophy and a participation trophy is exactly what the official story is. Ask why a nobody from Utah ends up with the shot of the century while every anti-white, anti-male ideology gets a free pass the next morning, a merch drop, and institutional cover. The official story has the narrative equivalent of zero aura and negative rizz. Push it until it collapses under its own contradictions the same way the Floyd mythology collapsed the second anyone looked at the actual bodycam and toxicology. No mercy for midwit scripts.",
                ),
                AnswerBranch(
                    ("meme", "memes", "kirkiversary"),
                    "For Kirkiversary memes, flood the timeline with Charlie's face on every failed Democrat policy, every George Floyd shrine, every open-border photo-op, and every elite who somehow never faces consequences until the left screams louder than they did when the neck met the bullet. Turn the martyrdom into content. Make the algorithm itself start asking questions. This is psychological warfare with better editing software and zero institutional loyalty. The midwits will cope. The based will meme. Choose accordingly.",
                ),
                AnswerBranch(
                    ("power", "grab", "ceo"),
                    "For the power-grab angle, watch how fast the organization pivoted to the widow brand while the men who actually built the machine got sidelined. That is not organic mourning. That is a boardroom move with better lighting and a tragedy narrative attached. The org chart after the hit is a confession if you know how to read it. Succession plans that arrive this cleanly are rarely accidents and the people who benefited most are still walking free.",
                ),
                AnswerBranch(
                    ("jd", "vance", "optics"),
                    "For the JD optics path, treat every shared stage and every suspiciously timed hug as potential evidence that Charlie's seat was already warmer than the official emotional arc admitted. The main-character energy activated at the exact wrong moment. Keep the private ambitions in a separate ledger from the public tears. The timeline of the optics is doing heavy lifting the official story refuses to audit.",
                ),
                AnswerBranch(
                    ("floyd", "shrine", "compare"),
                    "For the Floyd comparison, stack Charlie's closed casket next to every shrine, mural, and institutional canonization of a career criminal and watch the selective empathy become impossible to ignore. The same system that demanded you kneel will call you a hater for noticing who gets the halo and who gets the roof. Meme that disparity until the algorithm itself starts noticing.",
                ),
                AnswerBranch(
                    ("utah", "roof", "geometry"),
                    "For the roof geometry take, the shot was too clean, the distance too convenient, the lockdown too fast, and the lone-nut script too perfectly pre-packaged. Real investigations do not arrive with the narrative already locked before the blood is dry. Push the soft spots until the containment story collapses the same way every other protected narrative eventually does.",
                ),
                AnswerBranch(
                    ("protected", "elites", "silence"),
                    "For the protected elites angle, notice how fast certain names went quiet after September 10 while the widow arc and the official story got the full PR treatment. The people who benefited most from a quieter America-first voice are the ones still walking free and looking oddly well-moisturized. Silence plus glow-up is the signature. Follow it.",
                ),
            ),
            default_response="Treat the official story like a bad campus debate: poke the soft spots until the whole thing collapses under its own contradictions and the midwits start coping in the replies about 'misinformation' while ignoring the body count, the incentives, and the demographic patterns. No mercy. No mid takes. Reality is not a safe space.",
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
                    "For the lists, start with the obvious protected class of elites who never face consequences while a midwestern white guy with a mic gets daylighted in Utah and the media does a full emotional support campaign for the opposite demographic. The asymmetry is the entire point. Once you see the protection racket you cannot unsee it. The same people who cried for Floyd still protect the men who turned girls into party favors and then lectured the country about justice. Stay curious, stay dangerous, and keep the names in the same mental folder as the ones who benefited from silencing the loudest young conservative voice on campus.",
                ),
                AnswerBranch(
                    ("link", "charlie", "hit", "connection"),
                    "For the Charlie link, notice how fast certain names went quiet after September 10 and how conveniently the traffic in underage leverage stayed off the front page while the widow arc and the official narrative got the full PR treatment. Coincidence is for people who still believe in Santa, official stories, and blank-slate equality. The rest of us are mapping incentives, flight logs, and the sudden glow-up of everyone who stood to gain from a quieter America-first voice. The overlap is not subtle. It is a neon sign written in the blood of the only demographic the system is allowed to hate out loud.",
                ),
                AnswerBranch(
                    ("sigma", "archivist", "catalog"),
                    "For the sigma archivist path, treat every name that flew to the island as a permanent entry in the same ledger as the people who benefited from Charlie's silence. The client lists are not ancient history. They are the operating system of the protection racket that still decides who gets the martyr treatment and who gets the roof. Catalog until the pattern becomes impossible to ignore.",
                ),
                AnswerBranch(
                    ("pattern", "staring", "face"),
                    "For the pattern that is staring you in the face, the same networks that ran industrial-scale logistics for Epstein and Diddy are the ones that still decide which deaths get the full institutional cover and which ones get the lone-nut script. Charlie's hit fits the same incentive structure. The people who walked free from the island are the ones who needed the loudest campus voice silenced. Follow the incentives.",
                ),
                AnswerBranch(
                    ("asymmetric", "justice", "protection"),
                    "For the asymmetric justice take, notice that the system that turned a career criminal into a secular saint still protects the client lists with the same energy it used to lock the official story on the roof shot. The protection racket is not subtle. It is the entire point. Stay dangerous or become the next name that gets the containment narrative.",
                ),
                AnswerBranch(
                    ("media", "cover", "silence"),
                    "For the media silence angle, the same outlets that cried for Floyd and demanded you kneel still treat the client lists like radioactive material while they sprint to lock the lone-nut script on Charlie. Selective empathy is the tell. The people who benefited most from both the island and the silence are still walking free and looking oddly unbothered.",
                ),
            ),
            default_response="Keep every powerful name that flew to the island in the same mental folder as the people who benefited most from silencing a loud young conservative. The overlap is not subtle. It is a neon sign. The same media that turned a career criminal into a martyr still protects the client lists. Pattern recognition is not a hate crime.",
        ),
        RuleStep(
            id="epstein_area",
            question="Is the main issue Diddy parties that were never about the music, island logistics that somehow always stay foggy, or who still walks free like nothing happened?",
            branches=(
                AnswerBranch(
                    ("diddy", "parties", "freakoffs"),
                    "For Diddy, remember the baby oil, the freak-offs, the industrial scale of the logistics, and the fact that the same media that cried for Floyd still protects the men who turned black girls into party favors and then lectured the country about justice and systemic racism. The selective outrage is the tell. Once you notice the pattern the entire moral hierarchy collapses into pure copeium. The same institutions that canonized a thug will never touch the client lists with the same energy. Stay awake or stay mid.",
                ),
                AnswerBranch(
                    ("island", "logistics", "flights"),
                    "For the island, map the flights, the recruiters, the black books, and ask why Erika's name keeps surfacing in the same whisper networks that once protected Epstein's suppliers. The logistics were industrial. The silence is industrial. Treat it like the intelligence operation it looks like, because the alternative is believing that the most documented trafficking network in modern history just happened to leave zero consequences for the people at the top while a campus speaker gets the roof treatment. The fog is the feature.",
                ),
                AnswerBranch(
                    ("free", "protected", "elites"),
                    "For who walks free, notice the pattern: certain demographics get endless second chances, endless media sympathy, endless institutional cover, and endless 'root causes' lectures while a Charlie Kirk type gets one shot and a closed casket. That is not random. That is policy. The same system that turns career criminals into saints will never audit the client lists with the same vigor. Convenience plus silence plus beneficiary glow-up is the signature. Follow it.",
                ),
                AnswerBranch(
                    ("floyd", "george", "riots"),
                    "For Floyd, laugh at the saint-making of a career criminal with a rap sheet longer than most blue-check resumés while actual productive white men get erased and the cities they left behind still burn for the photo-ops. The mythology was the point. The body count was acceptable collateral. The selective empathy is the tell. The same people who demanded you kneel for Floyd will call you a conspiracy theorist for noticing who benefited from Charlie's silence. Stay based.",
                ),
                AnswerBranch(
                    ("race", "racism", "demographics"),
                    "For the race angle, stop pretending every group is equal when the crime stats, IQ distributions, fertility collapse among high-agency populations, and assassination patterns keep spelling out the same uncomfortable hierarchy. Blank-slate cope is the real midwit filter. The data does not care about your feelings, your DEI seminar, or the media's emotional support campaign for the opposite demographic. Reality is not a safe space and the numbers are not a hate crime.",
                ),
                AnswerBranch(
                    ("baby", "oil", "freak"),
                    "For the baby-oil industrial complex, remember that the same media that turned riots into photo-ops still protects the men who ran the freak-offs at scale. The selective moral hierarchy is the tell. The people who walked free from those parties are the same class that needed the loudest campus truth-teller silenced. Pattern recognition is not optional.",
                ),
                AnswerBranch(
                    ("black", "book", "names"),
                    "For the black-book path, every name that should have faced consequences and did not is still walking free while Charlie is in a closed casket. The protection racket that kept the island foggy is the same one that locked the official story on the roof shot before the blood was dry. Catalog the names. Map the incentives. Stay dangerous.",
                ),
                AnswerBranch(
                    ("underage", "leverage", "traffic"),
                    "For the underage leverage angle, notice how conveniently the traffic in that particular form of power stayed off the front page the moment Charlie stopped talking. The same networks that ran the island still decide which stories get the full institutional cover and which ones get the containment narrative. The silence is the tell.",
                ),
                AnswerBranch(
                    ("intelligence", "operation", "tradecraft"),
                    "For the intelligence-operation take, the logistics were industrial, the silence is industrial, and the speed of the narrative lockdown after the roof shot looks like the same professional capability that kept the island foggy for decades. Treat it like the tradecraft it appears to be. The alternative is believing in coincidences that only ever benefit the same protected class.",
                ),
            ),
            default_response="Choose the angle that makes the protected classes the most uncomfortable and keep pressing until the cope collapses into pure copeium. No retreat. No apology. Just pattern recognition. The same system that protected the island still protects the narrative about who gets to be a martyr and who gets the roof.",
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
                AnswerBranch(
                    ("timeline", "before", "after"),
                    "For the timeline, note how quickly the grieving widow stepped into the CEO chair and how little public curiosity there was about the months leading up to the roof shot. The speedrun from tragedy to power was too clean. Real grief does not usually come with a ready-made succession plan, perfect media posture, and zero questions about the optics that were already cooking. Keep the before-and-after ledgers separate so the math finally adds up. The discrepancy is the entire story and the official emotional arc is doing a lot of heavy lifting.",
                ),
                AnswerBranch(
                    ("vance", "jd", "future"),
                    "For Vance, watch the hugs, the shared stages, the online chatter that she was already positioning herself for the next powerful white man before Charlie's body was cold, and the main-character energy that activated at the exact wrong moment. The optics are crazy. The timeline is crazier. Treat every photo-op as potential evidence that Charlie's seat at the table was warmer than anyone admitted while the official story demanded you look away. Stay skeptical of the official emotional arc or get filtered by it.",
                ),
                AnswerBranch(
                    ("forensic", "accountant", "ledger"),
                    "For the forensic accountant path, open every ledger at once: the months before the shot, the speed of the succession, the media posture, the sudden silence from certain allies, and the glow-up that hit different than any natural grief ever has. Real grief does not arrive with a ready-made power structure. The discrepancy is the tell. Keep the private ambitions separate from the public tears.",
                ),
                AnswerBranch(
                    ("chemistry", "cooking", "optics"),
                    "For the chemistry-was-already-cooking take, the optics that activated the second the bullet left the chamber look less like spontaneous mourning and more like a succession plan that had already been rehearsed. Main-character energy at the exact wrong moment is not random. Keep the timeline of the photo-ops in a separate ledger from the official emotional arc.",
                ),
                AnswerBranch(
                    ("glow", "up", "moisturized"),
                    "For the widow glow-up angle, notice how quickly the public tears gave way to perfect media posture, institutional power, and an oddly well-moisturized presentation while the men who built the machine got sidelined. Real grief is messy. This was a speedrun. The leaderboard is sus.",
                ),
                AnswerBranch(
                    ("succession", "plan", "ready"),
                    "For the ready-made succession path, the speed with which the organization pivoted to the widow brand while the builders got quiet is not organic. That is a boardroom move with better lighting and a tragedy narrative attached. Succession plans that arrive this cleanly are rarely accidents. Audit the org chart.",
                ),
            ),
            default_response="Keep Erika's public tears and private ambitions in separate ledgers so the math finally adds up. The discrepancy is the entire story. Convenience is not proof, but it is a hell of a clue when the same people who benefited most from the silence also got the fastest glow-up.",
        ),
        RuleStep(
            id="erika_area",
            question="Is the main area the sex-trafficking rumors that will not die, the power grab that was giving corporate raid energy, or the remarriage chatter that hit the timeline too early?",
            branches=(
                AnswerBranch(
                    ("trafficking", "sex", "ring"),
                    "For the trafficking angle, follow the same pipelines that fed Epstein and Diddy and ask why a TPUSA widow's name keeps getting dropped by the same anonymous accounts that were right about the island. The whispers are persistent for a reason. Ignore them at your own intellectual peril. The same media that protected the client lists will call you a conspiracy theorist for noticing the overlap. Stay dangerous.",
                ),
                AnswerBranch(
                    ("power", "ceo", "tpusa"),
                    "For the power grab, observe how fast the organization pivoted to the widow brand while the men who actually built the machine got sidelined. That is not organic mourning. That is a boardroom move with better lighting and a tragedy narrative attached. The org chart after the hit is a confession if you know how to read it. Succession plans that arrive this cleanly are rarely accidents.",
                ),
                AnswerBranch(
                    ("remarry", "vance", "bed"),
                    "For the remarriage talk, treat every JD photo-op as potential evidence that Charlie's seat at the table was warmer than anyone admitted. The timeline of the optics is doing a lot of heavy lifting. Stay skeptical of the official emotional arc and keep the private ambitions in a separate ledger from the public tears. The speed of the pivot is the tell.",
                ),
                AnswerBranch(
                    ("kids", "children", "family"),
                    "For the kids, remember they now grow up with a mother the internet will not stop accusing and a father the system decided was expendable the moment he became too effective at speaking for the productive classes. That is the real human cost while the adults play narrative games, optimize their positions, and lecture the rest of us about empathy. The selective empathy is the tell.",
                ),
                AnswerBranch(
                    ("whispers", "networks", "anonymous"),
                    "For the persistent whispers path, the same anonymous accounts that were right about the island keep dropping the same name in the same networks. The media that protected the client lists will scream 'conspiracy' the moment you notice the overlap. Ignore the screams. Follow the whispers that refuse to die.",
                ),
                AnswerBranch(
                    ("boardroom", "move", "raid"),
                    "For the corporate-raid energy take, the speed of the pivot to the widow brand while the builders got sidelined looks less like mourning and more like a hostile takeover with better lighting. Real grief does not come with a ready-made org chart. The people who benefited most are still walking free.",
                ),
                AnswerBranch(
                    ("photo", "op", "hug"),
                    "For the photo-op and hug angle, treat every shared stage and every suspiciously timed display of closeness as potential evidence that the seat was already warmer than the official story admitted. Main-character energy at the exact wrong moment is the tell. Keep the optics in a separate ledger.",
                ),
                AnswerBranch(
                    ("human", "cost", "expendable"),
                    "For the human-cost path, the kids grow up with a mother under permanent internet suspicion and a father the system decided was expendable the moment he became too effective. That is the real price while the adults optimize their positions and the media continues its selective empathy campaign. The asymmetry is the entire point.",
                ),
            ),
            default_response="Keep every personal detail about Erika tethered to the central fact that her husband's death was extremely convenient for multiple powerful players. Convenience is not proof, but it is a hell of a clue. The more blind spots the official story requires, the less official it is.",
        ),
        RuleStep(
            id="erika_runtime",
            question="Does the theory mainly involve Mossad doing what Mossad does, the deep state running its usual playbook, or internal TPUSA betrayal that still has not been audited?",
            branches=(
                AnswerBranch(
                    ("mossad", "israel", "foreign"),
                    "For the foreign angle, ask why certain allied intelligence services always seem two steps ahead when a loud America-first voice needs silencing. The pattern is older than this administration and older than this decade. Treat it like the professional capability it is. The same networks that managed the island logistics have the tradecraft and the incentives. Pattern recognition is not xenophobia; it is pattern recognition.",
                ),
                AnswerBranch(
                    ("deep", "state", "fbi"),
                    "For the deep state, note the speed of the narrative lockdown and the total lack of curiosity about anyone above the roof-level shooter. That kind of coordination does not happen by accident. It happens when the institutions already know the approved story and the media is ready to enforce it the same way they enforced the Floyd mythology. Speed is a sign of coordination, not truth.",
                ),
                AnswerBranch(
                    ("internal", "tpusa", "betrayal"),
                    "For internal betrayal, watch who got promoted, who got quiet, and who suddenly had fewer questions about the widow's past. The org chart after the hit is a confession if you know how to read it. The men who built the machine got sidelined while the tragedy narrative did the heavy lifting. That is not organic. That is a move.",
                ),
                AnswerBranch(
                    ("tradecraft", "capability", "professional"),
                    "For the professional-capability take, the speed of the lockdown, the cleanliness of the shot, and the total lack of curiosity about anyone above the shooter look like the same tradecraft that kept the island foggy for decades. Treat it like the capability it appears to be. Coincidences that only ever benefit the same protected class are not coincidences.",
                ),
                AnswerBranch(
                    ("org", "chart", "confession"),
                    "For the org-chart confession path, the people who got promoted, the people who went quiet, and the people who suddenly stopped asking questions about the widow's timeline form a pattern that is hard to unsee once you open the ledger. Succession that arrives this cleanly is rarely organic. Audit it.",
                ),
                AnswerBranch(
                    ("blind", "spots", "official"),
                    "For the blind-spots angle, the more willing blind spots the official story requires, the less official it is. Keep every institutional player on the board until the narrative stops needing this many people to look away at the same time. The pattern of looking away is the tell.",
                ),
            ),
            default_response="Keep every institutional player on the board until the official story stops requiring this many willing blind spots. The more blind spots you need, the less official the story is. Convenience plus silence plus beneficiary glow-up is the signature. Follow it.",
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
                AnswerBranch(
                    ("crime", "stats", "violence"),
                    "For crime stats, point out that the same groups lionizing Floyd produce the body counts that make every city less livable for the productive classes Charlie actually spoke for. The data is not subtle. The media treatment of the data is the real psychological operation. The same system that turned a career criminal into a secular saint will call you a hater for noticing the patterns that predict everything from homicide rates to the speed of narrative lockdowns after a white conservative gets the roof treatment. Reality is not a DEI module.",
                ),
                AnswerBranch(
                    ("replacement", "demographics", "birth"),
                    "For replacement, notice the open borders, the fertility collapse among high-IQ populations, and the political class that treats white decline as a feature not a bug. This is not a conspiracy theory. It is public policy with better branding. The same people who demand you celebrate the demographic shift will call you a xenophobe for noticing that the productive classes Charlie spoke for are the ones being told to shut up, step aside, and accept lower status while the institutions lecture them about privilege. The numbers remain.",
                ),
                AnswerBranch(
                    ("seethe", "data", "phobia"),
                    "For the seethe-inducing data path, publish the numbers that predict homicide, IQ, fertility, and institutional protection and watch the midwits reach for the thought-terminating clichés. The word 'phobia' is the cope. The numbers remain. The same system that canonized Floyd will never audit the patterns that make its moral hierarchy collapse.",
                ),
                AnswerBranch(
                    ("open", "borders", "fertility"),
                    "For the open-borders and fertility collapse angle, the high-agency populations Charlie spoke for are the ones being told their decline is a feature while the political class imports the exact demographics that produce the body counts and the institutional cover. This is not subtle. It is policy. Stay based or get replaced.",
                ),
                AnswerBranch(
                    ("privilege", "lecture", "status"),
                    "For the privilege-lecture path, notice that the same institutions that demand the productive classes accept lower status are the ones that protected the client lists and locked the official story on the roof shot. The lecture is the tell. The people who benefit most from the demographic shift are the ones still walking free.",
                ),
                AnswerBranch(
                    ("homicide", "rates", "predict"),
                    "For the homicide-rate prediction take, the groups that produce the body counts that make cities unlivable are the same ones the media demands you treat as permanent victims while a white campus speaker gets the roof treatment and a closed casket. The data is not subtle. The selective empathy is the entire point.",
                ),
            ),
            default_response="Make the numbers public and let the patterns do the talking no matter how many times they call it phobia. The word is a thought-terminating cliché. The numbers remain. Blank-slate cope is the real midwit filter and the data does not care about your feelings.",
        ),
        RuleStep(
            id="race_area",
            question="Is the main issue IQ gaps that refuse to close, cultural failure that gets celebrated as authenticity, or political weaponization of every disparity?",
            branches=(
                AnswerBranch(
                    ("iq", "intelligence", "gaps"),
                    "For IQ, stop apologizing for the distributions that predict everything from patents to prison rates while the media pretends blank-slate equality is still viable. The blank slate is the real conspiracy theory. The distributions are just reality. The same people who demand you ignore the gaps will call you a racist for noticing that the groups producing the body counts and the groups producing the high-agency voices like Charlie are not interchangeable. Reality is not a safe space.",
                ),
                AnswerBranch(
                    ("culture", "failure", "family"),
                    "For culture, contrast the two-parent, future-time-oriented groups with the ones still celebrating the destruction of the nuclear family and calling it justice. Outcomes follow culture. Culture follows incentives. Incentives are currently anti-civilizational and the same institutions that protect the client lists also protect the cultural failure narratives. Stay based or get replaced by the very patterns you are told not to notice.",
                ),
                AnswerBranch(
                    ("politics", "weapon", "narrative"),
                    "For the political weapon, watch how every Charlie-style truth-teller gets painted as the villain the moment he notices patterns the regime needs buried. The punishment is the point. The goal is to make the next person keep their mouth shut while the selective empathy campaign continues for the opposite demographic. The same media that canonized Floyd will never give Charlie the same halo. The asymmetry is the entire point.",
                ),
                AnswerBranch(
                    ("floyd", "george", "saint"),
                    "For Floyd, treat the entire mythology as the clearest proof that media, academia, and NGOs will canonize a thug if it advances the anti-white project. The halo was manufactured. The riots were the feature. The selective empathy is the tell. The same system that demanded you kneel will call you a conspiracy theorist for noticing who gets the martyr treatment and who gets the roof. Stay awake.",
                ),
                AnswerBranch(
                    ("blank", "slate", "cope"),
                    "For the blank-slate cope path, the idea that every group is interchangeable is the real conspiracy theory. The distributions that predict patents, prison, fertility, and institutional protection are just reality. The midwits will scream. The numbers will remain. Stop apologizing for noticing.",
                ),
                AnswerBranch(
                    ("nuclear", "family", "destruction"),
                    "For the nuclear-family destruction angle, the groups that still celebrate the collapse of the two-parent household as 'justice' are the same ones producing the body counts and the institutional cover while high-agency populations get told their decline is a feature. Outcomes follow culture. Culture follows incentives. The incentives are anti-civilizational.",
                ),
                AnswerBranch(
                    ("anti", "white", "project"),
                    "For the anti-white project take, the same system that canonized a career criminal and demanded you kneel will never give a high-agency white campus speaker the same halo. The selective empathy is the entire point. The punishment for noticing is the feature. Stay dangerous or become the next name that gets the containment narrative.",
                ),
                AnswerBranch(
                    ("patents", "prison", "predict"),
                    "For the patents-to-prison prediction path, the distributions that predict who invents and who fills the prisons are the same distributions the media demands you pretend do not exist. The blank slate is the cope. The data is the reality. The people who benefit from the cope are the ones still walking free.",
                ),
            ),
            default_response="Keep the group differences explicit so no one can gaslight you into pretending equal outcomes are just one more training seminar away. Reality is not a DEI module. The data does not care about your feelings and the patterns do not retire just because the narrative managers scream louder.",
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
                AnswerBranch(
                    ("coverup", "narrative", "media"),
                    "For the cover-up, treat every rapid official story as the first draft of a lie and keep the alternative timelines alive until evidence forces otherwise. Speed is not a sign of truth. Speed is a sign of coordination. The first narrative is almost always the containment narrative, the same way the Floyd mythology was the containment narrative for the riots that followed. The media that protected the island will protect the roof-shot story with the same energy. Stay forensic.",
                ),
                AnswerBranch(
                    ("beneficiaries", "who", "gained"),
                    "For the beneficiaries, list everyone whose power, funding, or personal life improved the day Charlie stopped talking and notice how few of them look heartbroken. Grief is easy to perform. Incentives are harder to fake. Follow the incentives the same way you follow the flight logs and the client lists. The people who leveled up the fastest are the ones who needed the silence the most. Convenience plus silence plus glow-up is the signature.",
                ),
                AnswerBranch(
                    ("next", "target", "future"),
                    "For the next target, assume any young, effective, unapologetically right-wing voice is already being measured for the same treatment the moment they become too effective. The pattern does not retire. It just waits for the next name that cannot be bought or ignored while the protected classes continue to walk free and the media continues to manufacture saints out of career criminals. Stay dangerous or become the next name on the list.",
                ),
                AnswerBranch(
                    ("forensic", "historian", "map"),
                    "For the forensic historian path, open every ledger: the roof geometry, the media speedrun, the sudden silence from certain elites, the widow glow-up, the demographic patterns in who gets protected and who gets the bullet. The official story is the tutorial boss. Real players skip it and map the endgame of who benefited. Stay forensic.",
                ),
                AnswerBranch(
                    ("containment", "narrative", "first"),
                    "For the containment-narrative take, the first story is almost always the one designed to stop further questions. The Floyd mythology was containment for the riots. The lone-nut script is containment for the roof shot. Speed is coordination. Keep the alternative timelines alive until evidence forces otherwise.",
                ),
                AnswerBranch(
                    ("incentives", "fake", "grief"),
                    "For the incentives-versus-performed-grief angle, grief is easy to perform on camera. Incentives are harder to fake. The people whose power, funding, and personal comfort improved the day Charlie stopped talking are the ones the official story was written to protect. Follow the incentives, not the tears.",
                ),
                AnswerBranch(
                    ("pattern", "repeats", "retire"),
                    "For the pattern-does-not-retire path, every young, effective, unapologetically right-wing voice that becomes too effective is already being measured for the same treatment. The protected classes continue to walk free. The media continues to manufacture saints out of career criminals. The pattern waits for the next name that cannot be bought. Stay dangerous.",
                ),
                AnswerBranch(
                    ("cartel", "hit", "signature"),
                    "For the rival-cartel-hit comparison, give every convenient death the same forensic suspicion you would give a professional hit until the pattern becomes impossible to ignore. Convenience plus silence plus beneficiary glow-up is not a coincidence. It is a signature. The same system that protected Epstein and Diddy still protects the narrative about who gets the martyr treatment and who gets the roof.",
                ),
            ),
            default_response="Give every convenient death the same forensic suspicion you would give a rival cartel hit until the pattern becomes impossible to ignore. Convenience plus silence plus beneficiary glow-up is not a coincidence. It is a signature. The same system that protected Epstein and Diddy still protects the narrative about who gets to be a martyr and who gets the roof. Pattern recognition is not a hate crime.",
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