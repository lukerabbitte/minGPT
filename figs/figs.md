### figs.md highlights

- 5.
- 6 achieves low loss quite well. Try its setup on bigger set?
- 7 shows too high a learning rate.
- 8 shows sensitivity even to a small amount more loss.
- 9 through 11 show buildup of same training run.
- 12 compares to 6 but has context length of 10. Worse.
- 13 is literally 12 but with context length back up to 30. (Wait, this is now same as 5).
- 14 uses more 'extreme' dataset, where 4s and 5s become 5s and 1s, 2s, and 3s become 1s in the star
  ratings. It has quite good performance in terms of loss coming down.
- 15 also uses more 'extreme' dataset - I took `goodreads_eval_modified.tsv` and split it 80/20. I then took the
  `goodreads_eval_modified_20pc.tsv` and trained on that. This guaranteed the same distribution of 1 and 5-star ratings between the 
  `goodreads_eval_modified.tsv` and the `goodreads_eval_modified_20pc.tsv`. `queries/rating_split.py`
  can be used to demonstrate this. The actual sampling was done by taking the user_id, item_id, rating,
  and timestep from the very last timestep for user 4. Of course, we could do a wider test on more users,
  but still it doesn't look too promising.
- 16 does the same with another user.
- Maybe to be accurate I need to do as an average across all users?

- Note that 1 - 16 were all for 'reward-only' architectures. After, we try re-including rtg and seeing the difference.