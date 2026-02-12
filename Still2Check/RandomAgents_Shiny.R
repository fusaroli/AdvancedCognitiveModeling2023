library(shiny)
library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)

# --- UI ---
ui <- fluidPage(
  titlePanel("Random Agent: Convergence & Variance Analysis"),
  
  sidebarLayout(
    sidebarPanel(
      # Simulation Parameters
      sliderInput("trials", 
                  "Number of Trials (Time steps):", 
                  min = 10, max = 500, value = 200, step = 10),
      
      sliderInput("n_sims", 
                  "Number of Agents (Simulations per rate):", 
                  min = 50, max = 2000, value = 500, step = 50),
      
      # Multi-select for rates
      checkboxGroupInput("selected_rates", 
                         "Select Rates to Compare:", 
                         choices = c("0.1", "0.3", "0.5", "0.7", "0.9"),
                         selected = c("0.1", "0.5", "0.9"),
                         inline = TRUE),
      
      br(),
      actionButton("run", "Run Simulation", class = "btn-primary", width = "100%"),
      hr(),
      helpText("Click 'Run Simulation' to update plots."),
      helpText("Note: Rate 0.5 has the highest variance (uncertainty). Rates near 0 or 1 converge faster.")
    ),
    
    mainPanel(
      # Plot 1: The Ribbons (Confidence Intervals)
      h4("1. Convergence of Estimates (Mean + 95% Interval)"),
      plotOutput("ribbonPlot", height = "350px"),
      
      br(),
      
      # Plot 2: The Variance
      h4("2. Variance Over Time"),
      plotOutput("varPlot", height = "350px")
    )
  )
)

# --- SERVER ---
server <- function(input, output) {
  
  # Reactive: Run Simulation only when button is pressed
  sim_data <- eventReactive(input$run, {
    
    # Show a progress bar while calculating
    withProgress(message = 'Simulating Agents...', value = 0, {
      
      # Get inputs
      rates <- as.numeric(input$selected_rates)
      n_sims <- input$n_sims
      trials <- input$trials
      
      # Function to run one rate (Vectorized for speed)
      run_rate <- function(r) {
        # 1. Create a Matrix: Rows = Trials, Cols = Simulations
        # This generates all coin flips at once
        mat <- matrix(rbinom(trials * n_sims, 1, r), nrow = trials, ncol = n_sims)
        
        # 2. Cumulative Mean (Sweep/apply is faster than loops)
        # We calculate cumsum for every column
        cum_mat <- apply(mat, 2, function(x) cumsum(x) / seq_along(x))
        
        # 3. Calculate Stats across rows (across agents for each time step)
        # Row Means = Average path
        means <- rowMeans(cum_mat)
        
        # Row Variance = Disagreement between agents
        vars <- apply(cum_mat, 1, var)
        
        # Row Quantiles = The "Ribbon" (Confidence Interval)
        # We grab the 2.5% and 97.5% boundaries
        lower <- apply(cum_mat, 1, quantile, probs = 0.025)
        upper <- apply(cum_mat, 1, quantile, probs = 0.975)
        
        # Return Tidy Data
        tibble(
          trial = 1:trials,
          rate_group = factor(r),
          mean_est = means,
          variance = vars,
          lower_bound = lower,
          upper_bound = upper
        )
      }
      
      # Run for all selected rates and combine
      # map_dfr loops through the rates and binds rows automatically
      final_df <- map_dfr(rates, run_rate)
      
      return(final_df)
    })
  }, ignoreNULL = FALSE) # Run once on startup automatically
  
  
  # --- PLOT 1: RIBBONS ---
  output$ribbonPlot <- renderPlot({
    req(sim_data())
    
    ggplot(sim_data(), aes(x = trial, y = mean_est, group = rate_group, fill = rate_group, color = rate_group)) +
      # The Uncertainty Ribbon
      geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound), alpha = 0.2, color = NA) +
      # The Mean Line
      geom_line(size = 1) +
      # Target Lines (dashed)
      geom_hline(yintercept = as.numeric(input$selected_rates), linetype = "dotted", alpha = 0.5) +
      theme_classic() +
      labs(y = "Cumulative Rate", x = "Trial Number", fill = "True Rate", color = "True Rate") +
      ylim(0, 1)
  })
  
  # --- PLOT 2: VARIANCE ---
  output$varPlot <- renderPlot({
    req(sim_data())
    
    # Create theoretical data for plotting dashed lines
    rates <- as.numeric(unique(sim_data()$rate_group))
    trials <- max(sim_data()$trial)
    
    
    ggplot() +
      # Empirical Variance (Solid Lines) from Simulation
      geom_line(data = sim_data(), 
                aes(x = trial, y = variance, color = rate_group), 
                size = 1) +
      
      theme_classic() +
      labs(y = "Variance", 
           x = "Trial Number", 
           color = "True Rate")
  })
}

shinyApp(ui, server)
