import {
  DEFAULT_RULES,
} from "./definitions";

import {
  RuleEngine,
} from "./engine";

export const browserRuleEngine =
  new RuleEngine(
    DEFAULT_RULES,
  );
