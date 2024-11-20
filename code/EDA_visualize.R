#neural data pre-process
pacman::p_load(rstudioapi, bruceR, dplyr,patchwork,rstatix,ggpubr,smplot2)

code_path <- getActiveDocumentContext()$path
set.wd(substr(code_path, 1, str_locate(code_path, "explore_spotify")[2]))
rm(list = ls())
#=================
#Trend Comparison
Data <- import("data/EDA_trending.csv") 
print(colnames(Data))
#=================
#Feeling
# Define metrics to plot
metrics <- c("valence", "danceability", "energy")
# Convert release_date to year and calculate mean/sd per year per artist
# Define some notable years and songs
db_notable_year <- c(1969,1983,1998,2017)  # Year of 1989 album
coldplay_notable_year <- c(2000,2008,2015,2020) # Year of Yellow release
oasis_notable_year <- c(1995,2002,2005,2010) # Year of (What's the Story) Morning Glory? release
radiohead_notable_year <- c(1993,1995,2001,2011) # Year of OK Computer release
Data %>%
  mutate(year = lubridate::year(as.Date(release_date))) %>%
  select(year, artist, name, all_of(metrics),popularity) %>%
  pivot_longer(cols = all_of(metrics), names_to = "metric", values_to = "value") %>%
  group_by(metric) %>%
  mutate(value = (value - min(value, na.rm=TRUE)) / (max(value, na.rm=TRUE) - min(value, na.rm=TRUE))) %>%  # Min-max normalization
  group_by(year, artist, metric) %>%
  summarise(
    name = name[which.max(popularity)],
    mean_val = mean(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    max_val=max(value, na.rm = TRUE),
    sd_val = sd(value, na.rm = TRUE)
  ) %>%
  ggplot(aes(x = year, y = mean_val, color = metric, fill = metric)) +
  geom_smooth(aes(group = metric), method = "loess", size = 1,alpha=0.3) +
  geom_point(size = 2) +
  #添加横向辅助线 0.5 0.75 0.25
  geom_hline(yintercept = c(0.5,0.75,0.25), linetype = "dashed", color = "gray") +
  facet_wrap(~artist,scales = "free_x") +
  # Add points for notable songs
  geom_text(data = . %>% filter(
    (artist == "David Bowie" & year %in% db_notable_year) |
    (artist == "Coldplay" & year %in% coldplay_notable_year) |
    (artist == "Oasis" & year %in% oasis_notable_year) |
    (artist == "Radiohead" & year %in% radiohead_notable_year)
  ), aes(y=0.5,label = name),color="black", size = 3, vjust = -1,angle=90) +
  theme_bruce() +
  labs(x = "Year", y = "Standardized Value", title = "Standardized Metrics Over Time by Britpop Artists") +
  scale_color_brewer(palette = "Set2") +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(limits=c(0,1),breaks=c(0,0.25,0.5,0.75,1),expand = expansion(mult = c(0.17, 0.17))) +
  theme(legend.position = "bottom")


#=================
#Mix
# Define metrics to plot
metrics <- c("speechiness", "acousticness", "instrumentalness")
# Convert release_date to year and calculate mean/sd per year per artist
# Define some notable years and songs
{
db_notable_year <- c(1969,1986,2002,2020)  # Year of 1989 album
coldplay_notable_year <- c(2000,2010,2017,2020) # Year of Yellow release
oasis_notable_year <- c(1995,2002,2005,2010) # Year of (What's the Story) Morning Glory? release
radiohead_notable_year <- c(1993,1997,2003,2011) # Year of OK Computer release
Data %>%
  mutate(year = lubridate::year(as.Date(release_date))) %>%
  select(year, artist, name, all_of(metrics),popularity) %>%
  pivot_longer(cols = all_of(metrics), names_to = "metric", values_to = "value") %>%
  group_by(metric) %>%
  mutate(value = (value - min(value, na.rm=TRUE)) / (max(value, na.rm=TRUE) - min(value, na.rm=TRUE))) %>%  # Min-max normalization
  group_by(year, artist, metric) %>%
  summarise(
    name = name[which.max(popularity)],
    mean_val = mean(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    max_val=max(value, na.rm = TRUE),
    sd_val = sd(value, na.rm = TRUE)
  ) %>%
  ggplot(aes(x = year, y = mean_val, color = metric, fill = metric)) +
  geom_smooth(aes(group = metric), method = "loess",se=F, size = 1,alpha=0.3) +
  geom_point(size = 2) +
  #添加横向辅助线 0.5 0.75 0.25
  geom_hline(yintercept = c(0.5,0.75,0.25), linetype = "dashed", color = "gray") +
  facet_wrap(~artist,scales = "free_x") +
  # Add points for notable songs
  geom_text(data = . %>% filter(
    (artist == "David Bowie" & year %in% db_notable_year) |
      (artist == "Coldplay" & year %in% coldplay_notable_year) |
      (artist == "Oasis" & year %in% oasis_notable_year) |
      (artist == "Radiohead" & year %in% radiohead_notable_year)
  ), aes(y=0.5,label = name),color="black", size = 3, vjust = -1,angle=90) +
  theme_bruce() +
  labs(x = "Year", y = "Standardized Value", title = "Standardized Metrics Over Time by Britpop Artists") +
  scale_color_brewer(palette = "Set2") +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(limits=c(0,1),breaks=c(0,0.25,0.5,0.75,1),expand = expansion(mult = c(0.17, 0.17))) +
  theme(legend.position = "bottom")
}

#=================
#Skill
# Define metrics to plot
metrics <- c("time_signature", "tempo", "key", "mode")
# Convert release_date to year and calculate mean/sd per year per artist
# Define some notable years and songs
{
  db_notable_year <- c(1969,1986,2002,2020)  # Year of 1989 album
  coldplay_notable_year <- c(2000,2010,2017,2020) # Year of Yellow release
  oasis_notable_year <- c(1995,2002,2005,2010) # Year of (What's the Story) Morning Glory? release
  radiohead_notable_year <- c(1993,1997,2003,2011) # Year of OK Computer release
  Data %>%
    mutate(year = lubridate::year(as.Date(release_date))) %>%
    select(year, artist, name, all_of(metrics),popularity) %>%
    pivot_longer(cols = all_of(metrics), names_to = "metric", values_to = "value") %>%
    group_by(metric) %>%
    mutate(value = (value - min(value, na.rm=TRUE)) / (max(value, na.rm=TRUE) - min(value, na.rm=TRUE))) %>%  # Min-max normalization
    group_by(year, artist, metric) %>%
    summarise(
      name = name[which.max(popularity)],
      mean_val = mean(value, na.rm = TRUE),
      min_val = min(value, na.rm = TRUE),
      max_val=max(value, na.rm = TRUE),
      sd_val = sd(value, na.rm = TRUE)
    ) %>%
    ggplot(aes(x = year, y = mean_val, color = metric, fill = metric)) +
    geom_smooth(aes(group = metric), method = "loess",se=F, size = 1,alpha=0.3) +
    geom_point(size = 2) +
    #添加横向辅助线 0.5 0.75 0.25
    geom_hline(yintercept = c(0.5,0.75,0.25), linetype = "dashed", color = "gray") +
    facet_wrap(~artist,scales = "free_x") +
    # Add points for notable songs
    geom_text(data = . %>% filter(
      (artist == "David Bowie" & year %in% db_notable_year) |
        (artist == "Coldplay" & year %in% coldplay_notable_year) |
        (artist == "Oasis" & year %in% oasis_notable_year) |
        (artist == "Radiohead" & year %in% radiohead_notable_year)
    ), aes(y=0.5,label = name),color="black", size = 3, vjust = -1,angle=90) +
    theme_bruce() +
    labs(x = "Year", y = "Standardized Value", title = "Standardized Metrics Over Time by Britpop Artists") +
    scale_color_brewer(palette = "Set2") +
    scale_fill_brewer(palette = "Set2") +
    scale_y_continuous(limits=c(0,1),breaks=c(0,0.25,0.5,0.75,1),expand = expansion(mult = c(0.17, 0.17))) +
    theme(legend.position = "bottom")
}


#=================
#Physical
# Define metrics to plot
metrics <- c("duration_ms", "loudness", "tempo")
# Convert release_date to year and calculate mean/sd per year per artist
# Define some notable years and songs
{
  db_notable_year <- c(1969,1986,2002,2020)  # Year of 1989 album
  coldplay_notable_year <- c(2000,2010,2017,2020) # Year of Yellow release
  oasis_notable_year <- c(1995,2002,2005,2010) # Year of (What's the Story) Morning Glory? release
  radiohead_notable_year <- c(1993,1997,2003,2011) # Year of OK Computer release
  Data %>%
    mutate(year = lubridate::year(as.Date(release_date))) %>%
    select(year, artist, name, all_of(metrics),popularity) %>%
    pivot_longer(cols = all_of(metrics), names_to = "metric", values_to = "value") %>%
    group_by(metric) %>%
    mutate(value = (value - min(value, na.rm=TRUE)) / (max(value, na.rm=TRUE) - min(value, na.rm=TRUE))) %>%  # Min-max normalization
    group_by(year, artist, metric) %>%
    summarise(
      name = name[which.max(popularity)],
      mean_val = mean(value, na.rm = TRUE),
      min_val = min(value, na.rm = TRUE),
      max_val=max(value, na.rm = TRUE),
      sd_val = sd(value, na.rm = TRUE)
    ) %>%
    ggplot(aes(x = year, y = mean_val, color = metric, fill = metric)) +
    geom_smooth(aes(group = metric), method = "loess",se=F, size = 1,alpha=0.3) +
    geom_point(size = 2) +
    #添加横向辅助线 0.5 0.75 0.25
    geom_hline(yintercept = c(0.5,0.75,0.25), linetype = "dashed", color = "gray") +
    facet_wrap(~artist,scales = "free_x") +
    # Add points for notable songs
    geom_text(data = . %>% filter(
      (artist == "David Bowie" & year %in% db_notable_year) |
        (artist == "Coldplay" & year %in% coldplay_notable_year) |
        (artist == "Oasis" & year %in% oasis_notable_year) |
        (artist == "Radiohead" & year %in% radiohead_notable_year)
    ), aes(y=0.5,label = name),color="black", size = 3, vjust = -1,angle=90) +
    theme_bruce() +
    labs(x = "Year", y = "Standardized Value", title = "Standardized Metrics Over Time by Britpop Artists") +
    scale_color_brewer(palette = "Set2") +
    scale_fill_brewer(palette = "Set2") +
    scale_y_continuous(limits=c(0,1),breaks=c(0,0.25,0.5,0.75,1),expand = expansion(mult = c(0.17, 0.17))) +
    theme(legend.position = "bottom")
}



Data %>% 
  filter(name %in% c("Yellow","Flags","Something Just Like This")) %>% 
  select(name,acousticness,danceability,energy,instrumentalness,liveness,loudness,speechiness,valence) %>% 
  .[1:3,]


Data %>%
  filter(artist == "Oasis") %>%
  arrange(desc(acousticness)) %>%
  select(name, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence)

#=================
#Trend Comparison
Data <- import("data/EDA_trending2.csv") 
print(colnames(Data))
#=================
#Feeling
# Define metrics to plot
metrics <- c("valence", "danceability", "energy")
# Convert release_date to year and calculate mean/sd per year per artist
# Define some notable years and songs
ts_notable_year <- c(2006,2012,2017,2020)  # Year of 1989 album
aurora_notable_year <- c(2015,2017,2020,2024) # Year of All Of Me
ladygaga_notable_year <- c(2008,2011,2017,2020) # Year of The Fame
avril_notable_year <- c(2002,2007,2013,2017) # Year of Let Go
Data %>%
  mutate(year = lubridate::year(as.Date(release_date))) %>%
  select(year, artist, name, all_of(metrics),popularity) %>%
  pivot_longer(cols = all_of(metrics), names_to = "metric", values_to = "value") %>%
  group_by(metric) %>%
  mutate(value = (value - min(value, na.rm=TRUE)) / (max(value, na.rm=TRUE) - min(value, na.rm=TRUE))) %>%  # Min-max normalization
  group_by(year, artist, metric) %>%
  summarise(
    name = name[which.max(popularity)],
    mean_val = mean(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    max_val=max(value, na.rm = TRUE),
    sd_val = sd(value, na.rm = TRUE)
  ) %>%
  ggplot(aes(x = year, y = mean_val, color = metric, fill = metric)) +
  geom_smooth(aes(group = metric), method = "loess", size = 1,alpha=0.3) +
  geom_point(size = 2) +
  #添加横向辅助线 0.5 0.75 0.25
  geom_hline(yintercept = c(0.5,0.75,0.25), linetype = "dashed", color = "gray") +
  facet_wrap(~artist,scales = "free_x") +
  # Add points for notable songs
  geom_text(data = . %>% filter(
    (artist == "Taylor Swift" & year %in% ts_notable_year) |
    (artist == "AURORA" & year %in% aurora_notable_year) |
    (artist == "Lady Gaga" & year %in% ladygaga_notable_year) |
    (artist == "Avril Lavigne" & year %in% avril_notable_year)
  ), aes(y=0.5,label = name),color="black", size = 3, vjust = -1,angle=90) +
  theme_bruce() +
  labs(x = "Year", y = "Standardized Value", title = "Standardized Metrics Over Time by Pop Artists") +
  scale_color_brewer(palette = "Set2") +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(limits=c(0,1),breaks=c(0,0.25,0.5,0.75,1),expand = expansion(mult = c(0.17, 0.17))) +
  theme(legend.position = "bottom")


#=================
#Mix
# Define metrics to plot
metrics <- c("speechiness", "acousticness", "instrumentalness")
# Convert release_date to year and calculate mean/sd per year per artist
# Define some notable years and songs
{
ts_notable_year <- c(2006,2012,2017,2020)  # Year of 1989 album
aurora_notable_year <- c(2015,2017,2020,2024) # Year of All Of Me
ladygaga_notable_year <- c(2008,2011,2017,2020) # Year of The Fame
avril_notable_year <- c(2002,2007,2013,2017) # Year of Let Go
Data %>%
  mutate(year = lubridate::year(as.Date(release_date))) %>%
  select(year, artist, name, all_of(metrics),popularity) %>%
  pivot_longer(cols = all_of(metrics), names_to = "metric", values_to = "value") %>%
  group_by(metric) %>%
  mutate(value = (value - min(value, na.rm=TRUE)) / (max(value, na.rm=TRUE) - min(value, na.rm=TRUE))) %>%  # Min-max normalization
  group_by(year, artist, metric) %>%
  summarise(
    name = name[which.max(popularity)],
    mean_val = mean(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    max_val=max(value, na.rm = TRUE),
    sd_val = sd(value, na.rm = TRUE)
  ) %>%
  ggplot(aes(x = year, y = mean_val, color = metric, fill = metric)) +
  geom_smooth(aes(group = metric), method = "loess",se=F, size = 1,alpha=0.3) +
  geom_point(size = 2) +
  #添加横向辅助线 0.5 0.75 0.25
  geom_hline(yintercept = c(0.5,0.75,0.25), linetype = "dashed", color = "gray") +
  facet_wrap(~artist,scales = "free_x") +
  # Add points for notable songs
  geom_text(data = . %>% filter(
    (artist == "David Bowie" & year %in% db_notable_year) |
      (artist == "AURORA" & year %in% aurora_notable_year) |
      (artist == "Lady Gaga" & year %in% ladygaga_notable_year) |
      (artist == "Avril Lavigne" & year %in% avril_notable_year)
  ), aes(y=0.5,label = name),color="black", size = 3, vjust = -1,angle=90) +
  theme_bruce() +
  labs(x = "Year", y = "Standardized Value", title = "Standardized Metrics Over Time by Pop Artists") +
  scale_color_brewer(palette = "Set2") +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(limits=c(0,1),breaks=c(0,0.25,0.5,0.75,1),expand = expansion(mult = c(0.17, 0.17))) +
  theme(legend.position = "bottom")
}

#=================
#Skill
# Define metrics to plot
metrics <- c("time_signature", "tempo", "key", "mode")
# Convert release_date to year and calculate mean/sd per year per artist
# Define some notable years and songs
{
  ts_notable_year <- c(2006,2012,2017,2020)  # Year of 1989 album
  aurora_notable_year <- c(2015,2017,2020,2024) # Year of All Of Me
  ladygaga_notable_year <- c(2008,2011,2017,2020) # Year of The Fame
  avril_notable_year <- c(2002,2007,2013,2017) # Year of Let Go
  Data %>%
    mutate(year = lubridate::year(as.Date(release_date))) %>%
    select(year, artist, name, all_of(metrics),popularity) %>%
    pivot_longer(cols = all_of(metrics), names_to = "metric", values_to = "value") %>%
    group_by(metric) %>%
    mutate(value = (value - min(value, na.rm=TRUE)) / (max(value, na.rm=TRUE) - min(value, na.rm=TRUE))) %>%  # Min-max normalization
    group_by(year, artist, metric) %>%
    summarise(
      name = name[which.max(popularity)],
      mean_val = mean(value, na.rm = TRUE),
      min_val = min(value, na.rm = TRUE),
      max_val=max(value, na.rm = TRUE),
      sd_val = sd(value, na.rm = TRUE)
    ) %>%
    ggplot(aes(x = year, y = mean_val, color = metric, fill = metric)) +
    geom_smooth(aes(group = metric), method = "loess",se=F, size = 1,alpha=0.3) +
    geom_point(size = 2) +
    #添加横向辅助线 0.5 0.75 0.25
    geom_hline(yintercept = c(0.5,0.75,0.25), linetype = "dashed", color = "gray") +
    facet_wrap(~artist,scales = "free_x") +
    # Add points for notable songs
    geom_text(data = . %>% filter(
      (artist == "Taylor Swift" & year %in% ts_notable_year) |
        (artist == "AURORA" & year %in% aurora_notable_year) |
        (artist == "Lady Gaga" & year %in% ladygaga_notable_year) |
        (artist == "Avril Lavigne" & year %in% avril_notable_year)
    ), aes(y=0.5,label = name),color="black", size = 3, vjust = -1,angle=90) +
    theme_bruce() +
    labs(x = "Year", y = "Standardized Value", title = "Standardized Metrics Over Time by Pop Artists") +
    scale_color_brewer(palette = "Set2") +
    scale_fill_brewer(palette = "Set2") +
    scale_y_continuous(limits=c(0,1),breaks=c(0,0.25,0.5,0.75,1),expand = expansion(mult = c(0.17, 0.17))) +
    theme(legend.position = "bottom")
}


Data %>%
  filter(artist == "Avril Lavigne") %>%
  arrange(desc(acousticness)) %>%
  select(name, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence)

