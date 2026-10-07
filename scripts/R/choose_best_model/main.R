library(readr); library(dplyr)

# Step 1: list files
root  <- getwd()
files <- list.files(root, pattern = "metrics\\.csv$", recursive = TRUE, full.names = TRUE)
cat("Files:", length(files), "\n")
cat("Total size MB:", round(sum(file.size(files)) / 1e6, 1), "\n")

# Step 2: read one file, time it
t <- system.time(d1 <- read_csv(files[1], show_col_types = FALSE, guess_max = 10000))
print(t); cat("Rows:", nrow(d1), "\n")
cat("Estimated total sec:", round(t["elapsed"] * length(files)), "\n")

# Step 3: read all, with progress and timing
dfs <- vector("list", length(files))
t <- system.time(
  for (i in seq_along(files)) {
    dfs[[i]] <- read_csv(files[i], show_col_types = FALSE, guess_max = 10000, progress = FALSE)
    if (i %% 10 == 0) cat(i, "/", length(files), "\n")
  }
)
print(t)

# Step 4: build labels per file (once, not per row), then bind
labels <- data.frame(
  source_file = sub(paste0("^", root, "/?"), "", files),
  subdir      = basename(dirname(files))
)
names(dfs) <- labels$source_file

t <- system.time(combined <- bind_rows(dfs, .id = "source_file"))
print(t)
combined <- combined %>% left_join(labels, by = "source_file") %>% relocate(source_file, subdir)
cat("Dim:", nrow(combined), "x", ncol(combined), "\n")
cat("Rows check:", nrow(combined) == sum(sapply(dfs, nrow)), "\n")

# Step 5: data handling
cat("Dim:", nrow(combined), "x", ncol(combined), "\n\n")
cat("== str ==\n"); str(combined, give.attr = FALSE, vec.len = 3)
combined$newspaper <- combined$subdir
combined$source_file <- NULL
combined$subdir <- NULL
combined$parse_failure_rows <- NULL
combined$config_id <- as.factor(combined$config_id)
combined$layout_mode <- as.factor(combined$layout_mode)
combined$detector <- as.factor(combined$detector)
combined$binarization <- as.factor(combined$binarization)
combined$ocr_extractor <- as.factor(combined$ocr_extractor)
combined$llm_extractor <- as.factor(combined$llm_extractor)
combined$newspaper <- as.factor(combined$newspaper)
cat("== str ==\n"); str(combined, give.attr = FALSE, vec.len = 3)

# Step 6: Add more data
sample_df <- read_csv("sample.csv")
sample_df$page_position <- as.factor(sample_df$page_position)
sample_df$scale <- as.factor(sample_df$scale)
sample_df$image_stem <- paste(sample_df$newspaper, sample_df$date, sep = "_")
sample_df$image_stem <- paste(sample_df$image_stem, sample_df$page_number, sep = "_")
combined <- left_join(combined, distinct(sample_df, image_stem, page_position, scale), by = "image_stem", relationship = "many-to-one")

# Step 7: Revise
combined$image_stem <- as.factor(combined$image_stem)
combined$detector <- factor(ifelse(is.na(combined$detector), "none", as.character(combined$detector)))
cat("\n== summary ==\n"); print(summary(combined))

# Step 8: Aggregate
pipe <- c("config_id", "detector", "binarization", "ocr_extractor", "llm_extractor")
ctx0 <- c("newspaper")
ctx1 <- c("newspaper", "page_position")
ctx2 <- c("newspaper", "page_position", "scale")

agg <- function(df, by) df %>%
  group_by(across(all_of(by))) %>%
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

add_tradeoff <- function(df) df %>%
  mutate(
    f1_per_sec = detection_f1 / total_pipeline_seconds,           # higher = better
    wer_x_sec  = matched_body_wer_mean * total_pipeline_seconds   # lower = better
  ) %>%
  arrange(desc(detection_f1), matched_body_wer_mean, total_pipeline_seconds)

agg_pipe <- add_tradeoff(agg(combined, pipe))
agg_ctx0 <- add_tradeoff(agg(combined, c(pipe, ctx0)))
agg_ctx1 <- add_tradeoff(agg(combined, c(pipe, ctx1)))
agg_ctx2 <- add_tradeoff(agg(combined, c(pipe, ctx2)))

ctxs <- list(ctx0 = ctx0, ctx1 = ctx1, ctx2 = ctx2)
aggl <- list(ctx0 = agg_ctx0, ctx1 = agg_ctx1, ctx2 = agg_ctx2)

top_n <- 5
top_fn <- function(df, by) df %>%
  group_by(across(all_of(by))) %>%
  arrange(desc(detection_f1), matched_body_wer_mean, total_pipeline_seconds, .by_group = TRUE) %>%
  slice_head(n = top_n) %>%
  mutate(rank = row_number()) %>%
  relocate(all_of(by), rank) %>%
  ungroup()

res <- list()
for (k in names(ctxs)) {
  res[[paste0("top_", k)]]  <- top_fn(aggl[[k]], ctxs[[k]])
  res[[paste0("best_", k)]] <- filter(res[[paste0("top_", k)]], rank == 1)
}
list2env(res, envir = .GlobalEnv)

# Save everything
dir.create("agg", showWarnings = FALSE)
all_out <- c(list(agg_pipe = agg_pipe), setNames(aggl, paste0("agg_", names(aggl))), res)
for (n in names(all_out)) {
  write_csv(all_out[[n]], file.path("agg", paste0(n, ".csv")))
  saveRDS(all_out[[n]], file.path("agg", paste0(n, ".rds")))
}

best_ctx0; best_ctx1; best_ctx2
