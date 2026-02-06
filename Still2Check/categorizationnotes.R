# --- Load Data ---
human_data_file <- "AlienData.txt" 
# Check if file exists before attempting to read
if (!file.exists(human_data_file)) {
  stop("Error: Human data file not found at path: ", human_data_file)
}
# Read the data, specifying delimiter and setting default column type to character 
# to avoid parsing issues initially.
human_raw_data <- read_delim(human_data_file, delim = ",", col_types = cols(.default = col_character())) %>% mutate(nutritious = nutricious)

# --- Initial Exploration ---
cat("Raw Data Structure:\n")
glimpse(human_raw_data)
cat("\nUnique Conditions:", paste(unique(human_raw_data$condition), collapse=", "), "\n") # Expect 1 (Group) and 2 (Individual)
cat("Unique Sessions:", paste(sort(unique(human_raw_data$session)), collapse=", "), "\n") # Expect 1, 2, 3
cat("Unique Subjects (raw):", length(unique(human_raw_data$subject)), "\n") # Count unique subject identifiers
cat("Trials per Subject per Session (approx):\n")
# Display counts to check for completeness
print(human_raw_data %>% count(condition, subject, session) %>% arrange(n))

# --- Basic Cleaning and Feature Extraction ---
# Convert relevant columns to numeric, handle potential NAs introduced by coercion
human_data_processed <- human_raw_data %>%
  mutate(
    # Convert metadata and response columns to numeric
    across(c(condition, subject, session, cycle, trial, test, category, response, 
             dangerous, nutritious, correct, cumulative, RT), 
           ~suppressWarnings(as.numeric(.))), # Suppress warnings for NAs by coercion
    
    # Extract 5 binary features from the stimulus filename (e.g., "11001.jpg")
    # Assuming filename format is consistent
    stimulus_str = str_extract(stimulus, "^[01]{5}"), # Extract first 5 binary digits
    f1 = as.numeric(str_sub(stimulus_str, 1, 1)),
    f2 = as.numeric(str_sub(stimulus_str, 2, 2)),
    f3 = as.numeric(str_sub(stimulus_str, 3, 3)),
    f4 = as.numeric(str_sub(stimulus_str, 4, 4)),
    f5 = as.numeric(str_sub(stimulus_str, 5, 5)),
    
    # Decode human response (1-4) into separate dangerous/nutritious choices
    # Response codes: 1=Peaceful/Non-Nutritious, 2=Peaceful/Nutritious, 
    #                 3=Dangerous/Non-Nutritious, 4=Dangerous/Nutritious
    response_dangerous = ifelse(response %in% c(3, 4), 1, 0),
    response_nutritious = ifelse(response %in% c(2, 4), 1, 0),
    
    # Ensure true categories are numeric 0/1 (already done during load, but double-check)
    true_dangerous = dangerous, 
    true_nutritious = nutritious,
    
    # Recalculate overall correctness based on the two decisions
    # (The 'correct' column in the raw data might be based on the combined 1-4 response)
    correct_dangerous = ifelse(response_dangerous == true_dangerous, 1, 0),
    correct_nutritious = ifelse(response_nutritious == true_nutritious, 1, 0),
    correct_overall = ifelse(correct_dangerous == 1 & correct_nutritious == 1, 1, 0),
    
    # Create unique subject ID across conditions
    # Condition 1: Group, Condition 2: Individual
    condition_label = factor(condition, levels=c(1, 2), labels=c("Group", "Individual")),
    # Combine condition and subject number for a unique ID
    unique_subject_id = paste(condition_label, subject, sep="_") 
    
  ) %>%
  # Filter out test trials (test == 1) if they exist, focus on learning trials (test == 0)
  # Assuming 'test' column indicates transfer phase (1) vs training (0)
  filter(test == 0) %>% 
  # Select relevant columns for analysis
  dplyr::select(
    condition_label, unique_subject_id, session, cycle, trial, 
    f1:f5, # Features
    true_dangerous, true_nutritious, # True labels
    response_dangerous, response_nutritious, # Human responses
    correct_dangerous, correct_nutritious, correct_overall # Correctness flags
  )

# --- Verify Structure ---
cat("\nProcessed Data Structure (Learning Trials Only):\n")
glimpse(human_data_processed)
cat("\nNumber of Unique Subjects/Groups:", n_distinct(human_data_processed$unique_subject_id), "\n")
cat("Trials per Subject/Group per Session (processed):\n")
# Check trial counts again after processing and filtering
trial_counts <- human_data_processed %>% count(unique_subject_id, session)
print(trial_counts)

# Check if trial counts match simulation (expecting 96 per session)
expected_trials <- 96
if (!all(trial_counts$n == expected_trials)) {
  warning("Trial counts per subject/session deviate from the expected ", expected_trials, ". Check filtering and data integrity.")
  print(trial_counts %>% filter(n != expected_trials))
}

# Check for NAs in critical columns after processing
cat("\nNA Counts in Key Columns:\n")
print(colSums(is.na(human_data_processed)))

# Filter data for session 3 learning trials
session3_data <- human_data_processed %>%
  filter(session == 3) %>%
  distinct(f1, f2, f3, f4, f5, true_nutritious) # Get unique stimuli and their labels

# Define the rules to test:
# Rule A: Sum >= 3 (from simulation setup/paper description)
rule_A <- function(f1, f2, f3, f4, f5) {
  ifelse((f1 + f2 + f3 + f4 + f5) >= 3, 1, 0)
}

# Rule B: Sum >= 3 OR the specific exception 01110
rule_B <- function(f1, f2, f3, f4, f5) {
  is_exception <- (f1 == 0 & f2 == 1 & f3 == 1 & f4 == 1 & f5 == 0)
  ifelse(((f1 + f2 + f3 + f4 + f5) >= 3) | is_exception, 1, 0)
}

# Rule C: Sum >= 4 OR the specific exception 01110 (derived directly from data labels)
rule_C <- function(f1, f2, f3, f4, f5) {
  is_exception <- (f1 == 0 & f2 == 1 & f3 == 1 & f4 == 1 & f5 == 0)
  ifelse(((f1 + f2 + f3 + f4 + f5) >= 4) | is_exception, 1, 0)
}

# Apply rules and calculate match rate (accuracy)
results <- session3_data %>%
  mutate(
    pred_A = rule_A(f1, f2, f3, f4, f5),
    pred_B = rule_B(f1, f2, f3, f4, f5),
    pred_C = rule_C(f1, f2, f3, f4, f5)
  )

cat("Match Rate for Session 3 Nutritious Rules vs. Data File Labels:\n")
cat("Rule A (Sum >= 3):", mean(results$pred_A == results$true_nutritious), "\n")
cat("Rule B (Sum >= 3 OR 01110):", mean(results$pred_B == results$true_nutritious), "\n")
cat("Rule C (Sum >= 4 OR 01110):", mean(results$pred_C == results$true_nutritious), "\n")

# Display cases where Rule A mismatches the data
cat("\nCases where Rule A (Sum >= 3) mismatches the data's 'nutricious' label:\n")
print(results %>% filter(pred_A != true_nutritious))
cat("\nCases where Rule B Sum >= 3 OR 01110) mismatches the data's 'nutricious' label:\n")
print(results %>% filter(pred_B != true_nutritious))
cat("\nCases where Rule C Sum >= 4 OR 01110) mismatches the data's 'nutricious' label:\n")
print(results %>% filter(pred_B != true_nutritious))
