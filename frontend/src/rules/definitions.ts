import type {
  ConversationRule,
} from "./models";

export const KIRK_RULE: ConversationRule = {
  id: "kirk",
  keywords: [
    "kirk",
    "charlie kirk",
    "charlie",
    "tpusa",
    "turning point",
  ],
  steps: [
    {
      id: "kirk_topic",
      question:
        "What would you like to discuss about Charlie Kirk: his background, campus debates, or political views?",
      branches: [
        {
          keywords: [
            "background",
            "biography",
            "life",
          ],
          response:
            "Charlie Kirk became known as the founder of Turning Point USA and as a prominent conservative political activist and commentator.",
        },
        {
          keywords: [
            "debates",
            "campus",
            "students",
          ],
          response:
            "Charlie Kirk became widely associated with campus appearances, political debates, and Turning Point USA's university-focused activism.",
        },
        {
          keywords: [
            "views",
            "politics",
            "political",
          ],
          response:
            "Charlie Kirk was associated with conservative politics and frequently discussed elections, culture, education, and public policy.",
        },
      ],
      defaultResponse:
        "I can discuss Charlie Kirk's public career, political activity, or campus work.",
    },

    {
      id: "kirk_detail",
      question:
        "Would you prefer a short summary or a chronological overview?",
      branches: [
        {
          keywords: [
            "short",
            "summary",
            "brief",
          ],
          response:
            "A short summary works: Kirk became a nationally known conservative activist through Turning Point USA, media appearances, and political campaigning.",
        },
        {
          keywords: [
            "chronological",
            "timeline",
            "history",
          ],
          response:
            "A chronological overview would follow his early activism, the growth of Turning Point USA, increasing media visibility, and later national political involvement.",
        },
      ],
      defaultResponse:
        "I can keep it concise or structure it as a timeline.",
    },
  ],
};

export const DEFAULT_RULES:
  readonly ConversationRule[] = [
    KIRK_RULE,
  ];
