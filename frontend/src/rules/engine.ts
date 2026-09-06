import type {
  ChatMessage,
} from "../types/chat";

import type {
  AnswerBranch,
  ConversationRule,
  RuleStep,
} from "./models";

export const NO_RULE_RESPONSE =
  "I don't have a predefined rule for that topic yet.";

function normalise(
  text: string,
): string {
  return text
    .toLocaleLowerCase()
    .trim()
    .replace(/\s+/g, " ");
}

function escapeRegExp(
  value: string,
): string {
  return value.replace(
    /[.*+?^${}()|[\]\\]/g,
    "\\$&",
  );
}

function containsKeyword(
  text: string,
  keyword: string,
): boolean {
  const normalisedText =
    normalise(text);

  const normalisedKeyword =
    normalise(keyword);

  if (!normalisedKeyword) {
    return false;
  }

  if (
    normalisedKeyword.includes(" ") ||
    normalisedKeyword.includes("-")
  ) {
    return normalisedText.includes(
      normalisedKeyword,
    );
  }

  const pattern =
    new RegExp(
      `(^|\\W)${escapeRegExp(
        normalisedKeyword,
      )}(?=\\W|$)`,
      "i",
    );

  return pattern.test(
    normalisedText,
  );
}

interface PendingStep {
  rule: ConversationRule;
  stepIndex: number;
}

export class RuleEngine {
  private readonly rules:
    readonly ConversationRule[];

  private readonly questionIndex =
    new Map<
      string,
      PendingStep
    >();

  constructor(
    rules: readonly ConversationRule[],
  ) {
    this.rules = rules;

    for (const rule of rules) {
      rule.steps.forEach(
        (step, stepIndex) => {
          if (
            this.questionIndex.has(
              step.question,
            )
          ) {
            throw new Error(
              `Rule questions must be unique: ${step.question}`,
            );
          }

          this.questionIndex.set(
            step.question,
            {
              rule,
              stepIndex,
            },
          );
        },
      );
    }
  }

  matchRule(
    text: string,
  ): ConversationRule | null {
    let bestRule:
      ConversationRule | null =
      null;

    let bestScore = 0;

    for (const rule of this.rules) {
      const score =
        rule.keywords.reduce(
          (total, keyword) =>
            total +
            (
              containsKeyword(
                text,
                keyword,
              )
                ? 1
                : 0
            ),
          0,
        );

      if (score > bestScore) {
        bestRule = rule;
        bestScore = score;
      }
    }

    return bestRule;
  }

  private pendingStep(
    messages: readonly ChatMessage[],
  ): PendingStep | null {
    for (
      let index =
        messages.length - 1;
      index >= 0;
      index -= 1
    ) {
      const message =
        messages[index];

      if (
        message.role === "user"
      ) {
        continue;
      }

      for (
        const [
          question,
          location,
        ] of
        this.questionIndex.entries()
      ) {
        if (
          message.content.includes(
            question,
          )
        ) {
          return location;
        }
      }

      return null;
    }

    return null;
  }

  private matchingBranch(
    step: RuleStep,
    userAnswer: string,
  ): AnswerBranch | null {
    for (
      const branch
      of step.branches
    ) {
      if (
        branch.keywords.some(
          (keyword) =>
            containsKeyword(
              userAnswer,
              keyword,
            ),
        )
      ) {
        return branch;
      }
    }

    return null;
  }

  private resolveStepAnswer(
    step: RuleStep,
    userAnswer: string,
  ): string {
    return (
      this.matchingBranch(
        step,
        userAnswer,
      )?.response ??
      step.defaultResponse
    );
  }

  private topicSwitch(
    currentRule:
      ConversationRule,
    currentStep: RuleStep,
    userText: string,
  ): ConversationRule | null {
    if (
      this.matchingBranch(
        currentStep,
        userText,
      )
    ) {
      return null;
    }

    const candidate =
      this.matchRule(userText);

    if (
      !candidate ||
      candidate.id ===
        currentRule.id
    ) {
      return null;
    }

    return candidate;
  }

  respond(
    messages:
      readonly ChatMessage[],
  ): string {
    if (
      messages.length === 0
    ) {
      throw new Error(
        "At least one chat message is required.",
      );
    }

    const latest =
      messages[
        messages.length - 1
      ];

    if (
      latest.role !== "user"
    ) {
      throw new Error(
        "The latest chat message must be from the user.",
      );
    }

    const priorMessages =
      messages.slice(0, -1);

    const pending =
      this.pendingStep(
        priorMessages,
      );

    if (pending) {
      const {
        rule,
        stepIndex,
      } = pending;

      const step =
        rule.steps[
          stepIndex
        ];

      const switchedRule =
        this.topicSwitch(
          rule,
          step,
          latest.content,
        );

      if (switchedRule) {
        return (
          switchedRule.steps[0]
            ?.question ??
          NO_RULE_RESPONSE
        );
      }

      const answer =
        this.resolveStepAnswer(
          step,
          latest.content,
        );

      const nextStep =
        rule.steps[
          stepIndex + 1
        ];

      if (nextStep) {
        return (
          `${answer}\n\n` +
          nextStep.question
        );
      }

      return answer;
    }

    const rule =
      this.matchRule(
        latest.content,
      );

    return (
      rule?.steps[0]
        ?.question ??
      NO_RULE_RESPONSE
    );
  }
}
