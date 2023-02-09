library(tidyverse)
d <- read_csv("~/Dropbox (Personal)/My courses/2022 - AdvancedCognitiveModeling/data/ToM_penny_2022-02-08 evening.csv") %>% 
  subset(!is.na(participant.label) & !is.na(player.tom_level)) %>%
  rename(
    ID = participant.label,
    BotStrategyN = player.tom_level,
    BotParameters = player.tom_state,
    Role = player.player_role,
    Trial = subsession.round_number,
    Choice = player.decision,
    BotChoice = player.tom_decision,
    Payoff = player.player_payoff,
    BotPayoff = player.tom_payoff
  )

d <- d %>% subset(ID !="Aarhus" & ID !="Dubrovnik")

d$participant._is_bot <- NULL
d$participant.id_in_session <- NULL
d$participant.code <- NULL
d$participant.mturk_worker_id <- NULL
d$participant.mturk_assignment_id <- NULL
d$session.code <- NULL
d$session.label <- NULL
d$session.experimenter_name <- NULL
d$session.mturk_HITId <- NULL
d$session.mturk_HITGroupId <- NULL
d$session.comment <- NULL
d$session.is_demo <- NULL
d$participant._index_in_pages <- NULL
d$participant._max_page_index <- NULL
d$participant._current_app_name <- NULL
d$participant._current_page_name <- NULL
d$participant.ip_address <- NULL
d$participant.time_started <- NULL
d$participant.visited <- NULL
d$participant.payoff <- NULL
d$player.id_in_group <- NULL
d$player.payoff <- NULL
d$group.id_in_subsession <- NULL

d$BotStrategy <- as.factor(d$BotStrategyN)

d$Trial <- ifelse(d$Trial>40, d$Trial - 40, d$Trial)
d$Trial <- ifelse(d$Trial>40, d$Trial - 40, d$Trial)
d$Trial <- ifelse(d$Trial>40, d$Trial - 40, d$Trial)
d$Trial <- ifelse(d$Trial>40, d$Trial - 40, d$Trial)
d$Trial <- ifelse(d$Trial>40, d$Trial - 40, d$Trial)

ggplot(d, aes(Trial, Payoff, group=BotStrategy, color=BotStrategy)) +geom_smooth(se=F) + theme_classic() + facet_wrap(.~Role)

d1 <- d %>% group_by(ID, BotStrategy) %>% dplyr::summarize(Score = sum(Payoff))

ggplot(d1, aes(BotStrategy, Score)) +
  geom_point(aes(color=ID)) +
  geom_boxplot(alpha=0.3) +
  theme_classic()

d2 <- d %>% group_by(ID) %>% dplyr::summarize(Score = sum(Payoff))

write_csv(d, "data/MP_MSc_CogSci22.csv")
