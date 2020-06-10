library(shiny)
library(ggplot2)
library(ggthemes)
library(shinythemes)


data <- read.csv('full_predictions.csv')

ui <- fluidPage(theme = shinytheme('paper'),
                
                titlePanel("Introducing LEBRON: Longevity Estimate Based on Recurrent Optimized Network"),
                
  
                tabsetPanel(
                  tabPanel("Introduction", fluid = TRUE,
                           
                           mainPanel(
                             h1('Methods'),
                             p("To create LEBRON, first we created 3 tree-based models to predict a player's All-NBA probability in a given season. We predicted All-NBA
                                probability for every player whose career started on or after the 1979-1980 season (introduction of the 3-point line) up until the 2018-19
                                season. LEBRON models a player's career arc based on these probabilities by using a deep learning sequence model (LSTM). At each step, we use
                                a player's entire All-NBA probability history to project their All-NBA probability for the following year. We continue this process until the
                                maximum career length in our data set (21 years).", style = "font-size:16px"),
                             br(),
                             p("This dashboard helps visualize LEBRON for each player. A full table of remaining All-NBA seasons for each player in our data set is available
                                in the original blog post. This is to help understand the year-by-year progression for each player. Note that only players who played at least 4
                                seasons as of the 2018-19 season are considered in LEBRON. So, there is no Embiid, Doncic, etc.", style = "font-size:16px"),
                             h1("Links"),
                             a(href="https://dribbleanalytics.blog/2020/06/lebron-longevity-metric/",
                               div("Click here to see the original blog post which includes a more detailed discussion of methods and results.", style = 'font-size:16px')),
                             br(),
                             a(href="https://github.com/dribbleanalytics/lebron-longeivty-metric/",
                               div("Click here to see the GitHub repository for the project which contains all code, data, and results.", style = 'font-size:16px'))
                             
                             )
                           ),
                  tabPanel("Visualization", fluid = TRUE,
                           sidebarLayout(
                             sidebarPanel(p("To view the LEBRON progression for any player in our data set, select a  player from the dropdown menus below.
                                             The graphs will automatically update. The dashed line indicates the player's season number as of the 2018-19 season (e.g.
                                             LeBron was in his 16th year, so there is a vertical line at x = 16).", style = 'font-size:16px'),
                                          br(),
                                          
                                          selectInput(inputId = "player1",
                                                      label = "Select player:",
                                                      choices = unique(data$player))
                                          
                             ),
                             mainPanel(plotOutput(outputId = "lebron_plot")
                                       )
                             )
                           )
                  )
                )



server <- function(input, output) {
  
  output$lebron_plot <- renderPlot({
    
    player_data <- data[which(data$player == input$player1),]
    curr_year <- player_data[[2]]
    seq_data <- player_data[c(seq(from = 4, to = 24, by = 1))]
    seq_range <- seq(from = 1, to = 21, by = 1)
    
    player_df <- data.frame(t(seq_data), seq_range)
    colnames(player_df) <- c("all_nba_prob", "year_num")
    rownames(player_df) <- NULL
    player_df['history'] <- ifelse(player_df['year_num'] <= curr_year, "red", "blue")
    
    p <- ggplot(player_df, aes(x = year_num, y = all_nba_prob, color = history)) + 
      geom_line(size = 2) +
      xlab('Year #') +
      ylab('P(All-NBA)') +
      geom_vline(xintercept = curr_year, color = 'black', size = 2, linetype = "dashed") +
      scale_color_manual(name = "", labels = c("Predicted", "Historical"), values = c("red", "blue")) + 
      ylim(0, 1) +
      ggtitle(paste("LEBRON remaining All-NBA probability projection:", round(player_data[[3]], 2))) +
      theme_light() +
      theme(plot.title = element_text(size=15))
    
    p
    
  })
  
}

shinyApp(ui = ui, server = server)
