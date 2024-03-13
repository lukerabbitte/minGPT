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
- 17 tries with yet another user.
- 18 is same as 17, but tries average building up over 30 recs rather than 10 at eval stage.

- Note that 1 - 16 were all for 'reward-only' architectures. After, we try re-including rtg and seeing the difference.

- 21 is the first time I feel like get_returns is properly working, and the code is better formatted too.
- 22 was a repeat of 21
- 23 was same as 22 with more epochs
- 24 was the same with just 10 epochs - bad results though
- 25 is back to 30 epochs - but with 30 recs per trajectory
- 26 is same as 25 but with `80pc` dataset - moar values
- 28 we take constant state ([1.]) and always compares to user_id 1 in the complete matrix (let
  this stand in as a sort of proxy for the whole group) num_recs
- 29 is same as 28, just to double check it worked
- 30 is same as 29, just added arg max
- 31 is same as 30
- 32 is same as 31, just added so returns-to-go during evaluation is always 275 (we're training on 55 interactions per user)
- 33 (dummy data) is same as 31 (ditch 55*5 evals and go back to 30) but now with less noisy eval -
  verify ratings against 50 users in eval dataset rather than just one!
- 34 is same as 33 but with actual data.
- 35 is same as 34 but with more epochs (50 instead of 30 - hoping to see less fluctuation over time)
  also now saving checkpoints per epoch.
- 36 is the same as 35 but only does 1 new recommendation per epoch
- 37 is same as 35 but back to (1, 2, 3, 4, 5) datasets.
- 38 tried but with 80pc rather than 20pc dataset.
- 39 is same as 37 but uses tracking sampler thing. Actually, this gives us an important insight - if
  loss doesn't decrease at all over training, the recommendations don't 'converge' at all over epochs.
- 40 is same as 36 where we only have 1 new recommendation per epoch, but it's trained on 80pc 
  dataset and we only do 30 epochs as there were signs of overfitting after that on experiment 36.
  It didn't perform well because we didn't update learning rate decay for larger dataset.
- 
- 41 some exps figs and checkpoints lost after Git reset.
  Trying the 1 recommendation again with 20pc dataset and learning rate decay. Suspect will b good.
- 42 same as 41.
- 43 back to basics 30 num_recs, 30 epochs, etc.
- 44 same as 43.

- `loss_and_action_loss_plot_with_info_1.svg` shows that for fixed data, the loss explodes when we only keep
  action embeddings and not state embeddings.
- `loss_and_action_loss_plot_with_info_2.svg` shows the same, except that we only kept return-to-go embedding
  instead of state embedding.