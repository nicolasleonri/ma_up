library(arrow)
library(dplyr)
library(stringr)
library(tidyr)

##### Functions

set.seed(42)
inspect_df <- function(df) {
  cat("=== DIMENSIONS ===\n")
  cat(sprintf("Rows: %d | Cols: %d\n\n", nrow(df), ncol(df)))
  
  cat("=== COLUMN TYPES ===\n")
  print(sapply(df, class))
  cat("\n")
  
  cat("=== SUMMARY ===\n")
  print(summary(df))
  cat("\n")
  print(str(df))
  cat("\n")
  
  cat("=== MISSING VALUES ===\n")
  na_counts <- colSums(is.na(df))
  na_pct <- round(na_counts / nrow(df) * 100, 1)
  print(data.frame(missing = na_counts, pct = na_pct)[na_counts > 0, , drop = FALSE])
  if (all(na_counts == 0)) cat("No missing values.\n")
  cat("\n")
  
  cat("=== FIRST 5 ROWS ===\n")
  print(head(df, 5))
}
clean_df <- function(df, url_col = "page_url") {
  df %>%
    mutate(
      newspaper = as.factor(newspaper),
      date = as.Date(date, "%Y-%m-%d"),
      edition = as.factor(edition),
      scale = as.integer(str_extract(.data[[url_col]], "(?<=scale=)\\d+")),
      scale = replace_na(scale, 100),
      scale = as.factor(scale),
      year = as.POSIXct(date, format = "%Y-%m-%d"),
      year = format(year, format="%Y"),
      year = as.factor(year),
    ) %>%
  group_by(newspaper, date) %>%
    mutate(
      last_page = max(page_number),
      page_position = case_when(
        page_number == 1              ~ "front",
        page_number == last_page      ~ "back",
        TRUE                          ~ "interior"
      ),
      page_position = as.factor(page_position),
    ) %>%
    ungroup() %>%
    select(-last_page)
}
sample_stratified <- function(df, strata_cols = NULL, n_per_newspaper = 10, seed = 42) {
  set.seed(seed)
  df %>%
    group_by(newspaper) %>%
    group_modify(~ {
      strata_counts <- .x %>%
        group_by(across(all_of(strata_cols))) %>%
        summarise(n = n(), .groups = "drop") %>%
        mutate(allocated = pmax(1, round(n / sum(n) * n_per_newspaper)))
      
      sampled <- .x %>%
        left_join(strata_counts, by = strata_cols) %>%
        group_by(across(all_of(strata_cols))) %>%
        group_modify(~ {
          k <- min(nrow(.x), unique(.x$allocated))
          slice_sample(.x, n = k)
        }) %>%
        ungroup() %>%
        select(-n, -allocated)
      
      slice_sample(sampled, n = min(nrow(sampled), n_per_newspaper))
    }) %>%
    ungroup()
}
clean_df_2 <- function(df) {
  df %>%
    mutate(
      newspaper = as.factor(newspaper),
      date = as.Date(date, "%Y-%m-%d"),
      edition = as.factor(edition),
      scale = as.factor(scale),
      year = format(year, format="%Y"),
      year = as.factor(year),
    ) %>%
    group_by(newspaper, date) %>%
    mutate(
      last_page = max(page_number),
      page_position = case_when(
        page_number == 1              ~ "front",
        page_number == last_page      ~ "back",
        TRUE                          ~ "interior"
      ),
      page_position = as.factor(page_position),
    ) %>%
    ungroup() %>%
    select(-last_page)
}

##### Extraction
df_missing_values <- read.csv2("missing_samples.csv", sep = ",")
df_missing_values <- clean_df_2(df_missing_values)
inspect_df(df_missing_values)
unique(df_missing_values$year)

df_correo <- arrow::read_parquet("correo_metadata.parquet")
df_correo <- clean_df(df_correo)
inspect_df(df_correo)
unique(df_correo$year)

df_elcomercio <- arrow::read_parquet("elcomercio_metadata.parquet")
df_elcomercio <- clean_df(df_elcomercio)
inspect_df(df_elcomercio)
unique(df_elcomercio$year)

df_gestion <- arrow::read_parquet("gestion_metadata.parquet")
df_gestion <- clean_df(df_gestion)
inspect_df(df_gestion)
unique(df_gestion$year)

df_ojo <- arrow::read_parquet("ojo_metadata.parquet")
df_ojo <- clean_df(df_ojo)
inspect_df(df_ojo)
unique(df_ojo$year)

df_peru21 <- arrow::read_parquet("peru21_metadata.parquet")
df_peru21 <- clean_df(df_peru21)
inspect_df(df_peru21)
unique(df_peru21$year)

df_trome <- arrow::read_parquet("trome_metadata.parquet")
df_trome <- clean_df(df_trome)
inspect_df(df_trome)
unique(df_trome$year)

##### Get samples
# unique((df_all[df_all$newspaper=="correo",]$year))
df_all <- bind_rows(df_correo, df_elcomercio, df_gestion, df_ojo, df_peru21, df_trome, df_missing_values)
df_sample <- sample_stratified(df_all, strata_cols = c("year", "page_position", "scale"), n_per_newspaper = 10, seed = 42)
inspect_df(df_sample)
write.csv2(df_sample, file="sample.csv")
