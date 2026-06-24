create extension if not exists "pgcrypto";

create table if not exists public.posts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  source_post_id bigint,
  caption text not null,
  image_path text,
  harmful_count integer not null default 0,
  model_label text not null default 'unknown',
  model_ensemble_probability double precision,
  model_multimodal_probability double precision,
  model_caption_probability double precision,
  created_at timestamptz not null default timezone('utc', now()),
  constraint posts_model_label_check
    check (model_label in ('safe', 'harmful', 'unknown'))
);

create index if not exists posts_created_at_idx on public.posts (created_at desc);
create index if not exists posts_user_id_idx on public.posts (user_id);

create table if not exists public.harmful_reactions (
  post_id uuid not null references public.posts (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  created_at timestamptz not null default timezone('utc', now()),
  primary key (post_id, user_id)
);

create or replace function public.sync_post_harmful_count()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  update public.posts
  set harmful_count = (
    select count(*)
    from public.harmful_reactions
    where post_id = coalesce(new.post_id, old.post_id)
  )
  where id = coalesce(new.post_id, old.post_id);

  return coalesce(new, old);
end;
$$;

drop trigger if exists harmful_reactions_sync_count on public.harmful_reactions;
create trigger harmful_reactions_sync_count
after insert or delete on public.harmful_reactions
for each row
execute function public.sync_post_harmful_count();

alter table public.posts enable row level security;
alter table public.harmful_reactions enable row level security;

drop policy if exists "Authenticated users can read posts" on public.posts;
create policy "Authenticated users can read posts"
on public.posts
for select
to authenticated
using (true);

drop policy if exists "Users can insert own posts" on public.posts;
create policy "Users can insert own posts"
on public.posts
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists "Users can update own posts" on public.posts;
create policy "Users can update own posts"
on public.posts
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "Users can delete own posts" on public.posts;
create policy "Users can delete own posts"
on public.posts
for delete
to authenticated
using (auth.uid() = user_id);

drop policy if exists "Authenticated users can read reactions" on public.harmful_reactions;
create policy "Authenticated users can read reactions"
on public.harmful_reactions
for select
to authenticated
using (true);

drop policy if exists "Users can create their reactions" on public.harmful_reactions;
create policy "Users can create their reactions"
on public.harmful_reactions
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists "Users can delete their reactions" on public.harmful_reactions;
create policy "Users can delete their reactions"
on public.harmful_reactions
for delete
to authenticated
using (auth.uid() = user_id);

insert into storage.buckets (id, name, public)
values ('post-images', 'post-images', true)
on conflict (id) do update
set public = excluded.public;

drop policy if exists "Public can view post images" on storage.objects;
create policy "Public can view post images"
on storage.objects
for select
to public
using (bucket_id = 'post-images');

drop policy if exists "Users can upload their post images" on storage.objects;
create policy "Users can upload their post images"
on storage.objects
for insert
to authenticated
with check (
  bucket_id = 'post-images'
  and position((auth.uid())::text || '/' in name) = 1
);

drop policy if exists "Users can update their post images" on storage.objects;
create policy "Users can update their post images"
on storage.objects
for update
to authenticated
using (
  bucket_id = 'post-images'
  and position((auth.uid())::text || '/' in name) = 1
)
with check (
  bucket_id = 'post-images'
  and position((auth.uid())::text || '/' in name) = 1
);

drop policy if exists "Users can delete their post images" on storage.objects;
create policy "Users can delete their post images"
on storage.objects
for delete
to authenticated
using (
  bucket_id = 'post-images'
  and position((auth.uid())::text || '/' in name) = 1
);
