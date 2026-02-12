library(shiny)
library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)

# --- UI ---
ui <- fluidPage(
  titlePanel("Noisy Agent: Impact of Noise on Convergence"),
  
  sidebarLayout(
    sidebarPanel(
      # 1. Simulation Size
      h4("1. Simulation Settings"),
      sliderInput("trials", "Trials (Time steps):", min = 10, max = 500, value = 300, step = 10),
      sliderInput("n_sims", "Agents per level:", min = 50, max = 2000, value = 1000, step = 50),
      
      hr(),
      
      # 2. Agent Parameters
      h4("2. Agent Parameters"),
      sliderInput("base_rate", 
                  "Base Rate (The Agent's Intended Goal):", 
                  min = 0, max = 1, value = 0.9, step = 0.1),
      
      checkboxGroupInput("selected_noise", 
                         "Select Noise Levels to Compare:", 
                         choices = c("0" = 0, "0.1" = 0.1, "0.3" = 0.3, "0.5" = 0.5, "1.0" = 1),
                         selected = c(0, 0.5),
                         inline = TRUE),
      
      br(),
      actionButton("run", "Run Simulation", class = "btn-primary", width = "100%"),
      
      hr(),
      helpText("Noise = 0: Agent follows Base Rate perfectly."),
      helpText("Noise = 1: Agent is completely random (50/50).")
    ),
    
    mainPanel(
      h4("1. Convergence (Ribbons = 95% Confidence)"),
      plotOutput("ribbonPlot", height = "350px"),
      
      br(),
      
      h4("2. Variance Over Time"),
      plotOutput("varPlot", height = "350px")
    )
  )
)

# --- SERVER ---
server <- function(input, output) {
  
  # Reactive: Run Simulation
  sim_data <- eventReactive(input$run, {
    withProgress(message = 'Simulating Noisy Agents...', value = 0, {
      
      base_rate <- input$base_rate
      noise_levels <- as.numeric(input$selected_noise)
      n_sims <- input$n_sims
      trials <- input$trials
      
      # Function to simulate one specific noise level
      run_noise_level <- function(noise_val) {
        
        # --- VECTORIZED NOISE LOGIC ---
        # 1. Generate the "Intended" choices (based on Base Rate)
        intended_choice <- rbinom(trials * n_sims, 1, base_rate)
        
        # 2. Generate the "Noise" triggers (1 = noise happens, 0 = signal okay)
        noise_trigger <- rbinom(trials * n_sims, 1, noise_val)
        
        # 3. Generate the "Random" noise choices (50/50 coin flips)
        random_choice <- rbinom(trials * n_sims, 1, 0.5)
        
        # 4. Combine: If noise triggered, use random_choice, else use intended_choice
        final_choices <- ifelse(noise_trigger == 1, random_choice, intended_choice)
        
        # Create matrix for stats calculation
        mat <- matrix(final_choices, nrow = trials, ncol = n_sims)
        
        # Calculate Cumulative Means (down columns)
        cum_mat <- apply(mat, 2, function(x) cumsum(x) / seq_along(x))
        
        # Calculate Stats (across rows/agents)
        tibble(
          trial = 1:trials,
          noise_group = factor(noise_val),
          # Effective Rate = Theoretical limit where it should converge
          # Formula: (Rate * (1-Noise)) + (0.5 * Noise)
          effective_target = (base_rate * (1 - noise_val)) + (0.5 * noise_val),
          
          mean_est = rowMeans(cum_mat),
          variance = apply(cum_mat, 1, var),
          lower = apply(cum_mat, 1, quantile, 0.025),
          upper = apply(cum_mat, 1, quantile, 0.975)
        )
      }
      
      # Run for all selected noise levels and combine
      map_dfr(noise_levels, run_noise_level)
    })
  }, ignoreNULL = FALSE)
  
  
  # --- PLOT 1: RIBBONS ---
  output$ribbonPlot <- renderPlot({
    req(sim_data())
    
    ggplot(sim_data(), aes(x = trial, y = mean_est, group = noise_group, fill = noise_group, color = noise_group)) +
      # Uncertainty Ribbons
      geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
      # Mean Lines
      geom_line(size = 1) +
      # Theoretical Convergence Lines (Dashed)
      geom_hline(aes(yintercept = effective_target, color = noise_group), 
                 linetype = "dashed", alpha = 0.7) +
      
      theme_classic() +
      ylim(0, 1) +
      labs(subtitle = paste("Base Rate =", input$base_rate, "| Dashed lines = Theoretical Convergence Points"),
           y = "Cumulative Rate", x = "Trial Number", 
           fill = "Noise Level", color = "Noise Level")
  })
  
  # --- PLOT 2: VARIANCE ---
  output$varPlot <- renderPlot({
    req(sim_data())
    
    ggplot(sim_data(), aes(x = trial, y = variance, color = noise_group)) +
      geom_line(size = 1) +
      theme_classic() +
      labs(subtitle = "Higher noise usually pushes variance higher (closer to the 0.5 random walk)",
           y = "Empirical Variance", x = "Trial Number", 
           color = "Noise Level")
  })
}

shinyApp(ui, server)
