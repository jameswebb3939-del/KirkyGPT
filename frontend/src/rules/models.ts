export interface AnswerBranch {
  keywords: readonly string[];
  response: string;
}

export interface RuleStep {
  id: string;
  question: string;
  branches: readonly AnswerBranch[];
  defaultResponse: string;
}

export interface ConversationRule {
  id: string;
  keywords: readonly string[];
  steps: readonly RuleStep[];
}
