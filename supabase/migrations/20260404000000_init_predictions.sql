-- IPL 2026 Predictor — Supabase Schema
-- Run this in the Supabase SQL Editor (Dashboard > SQL Editor > New query)

-- Predictions table (one row per prediction run)
create table if not exists predictions (
  id           bigserial primary key,
  user_id      uuid references auth.users(id) on delete cascade not null,
  created_at   timestamptz default now(),
  team1        text not null,
  team2        text not null,
  venue        text,
  toss_winner  text,
  toss_decision text,
  team1_xi     jsonb,
  team2_xi     jsonb,
  team1_impact text,
  team2_impact text,
  predicted_winner text,
  t1_prob      float,
  t2_prob      float
);

-- Row-Level Security: each user only sees and writes their own rows
alter table predictions enable row level security;

create policy "select own predictions"
  on predictions for select
  using (auth.uid() = user_id);

create policy "insert own predictions"
  on predictions for insert
  with check (auth.uid() = user_id);

create policy "delete own predictions"
  on predictions for delete
  using (auth.uid() = user_id);

-- Index for fast per-user history lookups
create index if not exists predictions_user_id_idx on predictions (user_id, created_at desc);
