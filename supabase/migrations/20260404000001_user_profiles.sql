-- User profiles table — one row per auth.users entry
create table if not exists user_profiles (
  id           uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  created_at   timestamptz default now()
);

-- RLS: users can only read/update their own profile
alter table user_profiles enable row level security;

create policy "select own profile"
  on user_profiles for select
  using (auth.uid() = id);

create policy "insert own profile"
  on user_profiles for insert
  with check (auth.uid() = id);

create policy "update own profile"
  on user_profiles for update
  using (auth.uid() = id);

-- Trigger: auto-create a blank profile row whenever a new user signs up
create or replace function handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.user_profiles (id, display_name)
  values (
    new.id,
    coalesce(new.raw_user_meta_data->>'display_name', split_part(new.email, '@', 1))
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure handle_new_user();
