import { useEffect, useMemo, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import type { AuthChangeEvent, Session } from '@supabase/supabase-js'
import './App.css'
import seiveLogo from './assets/seive-logo.png'
import { type ModerationVerdict, classifyPostBeforePublish } from './lib/moderation'
import { deletePostViaBackend } from './lib/posts'
import { POSTS_BUCKET, isSupabaseConfigured, supabase } from './lib/supabase'

type ModelLabel = 'safe' | 'harmful' | 'unknown'

type FeedPost = {
  id: string
  user_id: string
  caption: string
  image_path: string | null
  created_at: string
  model_label?: ModelLabel | null
  model_ensemble_probability?: number | null
  model_multimodal_probability?: number | null
  model_caption_probability?: number | null
}

type LocalModelOverride = {
  modelLabel: ModelLabel
  modelEnsembleProbability: number | null
  modelMultimodalProbability: number | null
  modelCaptionProbability: number | null
}

type FeedbackSelection = 'safe' | 'unsafe'
type HarmfulReactionRow = { post_id: string }

const BASIC_POST_SELECT = 'id,user_id,caption,image_path,created_at'
const MODEL_POST_SELECT = `${BASIC_POST_SELECT},model_label,model_ensemble_probability,model_multimodal_probability,model_caption_probability`

function isMissingModelColumnError(message: string) {
  const lower = message.toLowerCase()
  return (
    lower.includes('model_label') ||
    lower.includes('model_ensemble_probability') ||
    lower.includes('model_multimodal_probability') ||
    lower.includes('model_caption_probability')
  )
}

function toModelLabel(verdict: ModerationVerdict | null): ModelLabel {
  if (!verdict) {
    return 'unknown'
  }
  return verdict.harmful ? 'harmful' : 'safe'
}

function buildModelOverride(verdict: ModerationVerdict | null): LocalModelOverride {
  return {
    modelLabel: toModelLabel(verdict),
    modelEnsembleProbability: verdict?.ensembleProbability ?? null,
    modelMultimodalProbability: verdict?.multimodalProbability ?? null,
    modelCaptionProbability: verdict?.captionProbability ?? null,
  }
}

function getDisplayName(session: Session | null) {
  const email = session?.user.email ?? 'Seive user'
  const [name] = email.split('@')
  return name || email
}

function getAvatarInitial(session: Session | null) {
  return getDisplayName(session).slice(0, 1).toUpperCase()
}

function formatPostTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return 'Just now'
  }
  return new Intl.DateTimeFormat('en', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

function getFileExtension(file: File) {
  const pieces = file.name.split('.')
  const lastPiece = pieces[pieces.length - 1]
  return lastPiece ? lastPiece.toLowerCase() : 'png'
}

function createUploadPath(session: Session, file: File) {
  const fileId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}`
  return `${session.user.id}/${Date.now()}-${fileId}.${getFileExtension(file)}`
}

function getImageUrl(imagePath: string | null) {
  if (!imagePath || !supabase) {
    return null
  }
  const { data } = supabase.storage.from(POSTS_BUCKET).getPublicUrl(imagePath)
  return data.publicUrl
}

function getModelState(
  post: FeedPost,
  localModelOverrides: Record<string, LocalModelOverride>,
): LocalModelOverride {
  const localOverride = localModelOverrides[post.id]
  if (localOverride) {
    return localOverride
  }

  return {
    modelLabel: post.model_label ?? 'unknown',
    modelEnsembleProbability: post.model_ensemble_probability ?? null,
    modelMultimodalProbability: post.model_multimodal_probability ?? null,
    modelCaptionProbability: post.model_caption_probability ?? null,
  }
}

function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin')
  const [authPending, setAuthPending] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const [posts, setPosts] = useState<FeedPost[]>([])
  const [loadingFeed, setLoadingFeed] = useState(false)
  const [supportsModelColumns, setSupportsModelColumns] = useState(true)
  const [localModelOverrides, setLocalModelOverrides] = useState<Record<string, LocalModelOverride>>(
    {},
  )
  const [feedbackSelections, setFeedbackSelections] = useState<Record<string, FeedbackSelection>>({})

  const [caption, setCaption] = useState('')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null)
  const [isComposerOpen, setIsComposerOpen] = useState(false)
  const [submitPending, setSubmitPending] = useState(false)
  const [moderationVerdict, setModerationVerdict] = useState<ModerationVerdict | null>(null)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [deletingPostId, setDeletingPostId] = useState<string | null>(null)
  const [reactionPendingPostId, setReactionPendingPostId] = useState<string | null>(null)

  useEffect(() => {
    if (!imageFile) {
      setImagePreviewUrl(null)
      return
    }

    const nextUrl = URL.createObjectURL(imageFile)
    setImagePreviewUrl(nextUrl)
    return () => URL.revokeObjectURL(nextUrl)
  }, [imageFile])

  useEffect(() => {
    if (!isComposerOpen) {
      return
    }

    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape' && !submitPending) {
        setIsComposerOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isComposerOpen, submitPending])

  useEffect(() => {
    const client = supabase
    if (!client) {
      setAuthLoading(false)
      return
    }

    let cancelled = false

    async function initialize() {
      const authClient = client as NonNullable<typeof client>
      const {
        data: { session: currentSession },
      } = await authClient.auth.getSession()

      if (cancelled) {
        return
      }

      setSession(currentSession)
      setAuthLoading(false)
      if (currentSession) {
        await loadFeed(currentSession)
      }
    }

    initialize().catch((error: unknown) => {
      if (cancelled) {
        return
      }
      setAuthLoading(false)
      setStatusMessage(error instanceof Error ? error.message : 'Unable to initialize Seive.')
    })

    const {
      data: { subscription },
    } = client.auth.onAuthStateChange(
      (_event: AuthChangeEvent, nextSession: Session | null) => {
        setSession(nextSession)
        setStatusMessage(null)
        if (nextSession) {
          void loadFeed(nextSession)
        } else {
          setPosts([])
          setFeedbackSelections({})
          setLocalModelOverrides({})
        }
      },
    )

    return () => {
      cancelled = true
      subscription.unsubscribe()
    }
  }, [])

  async function loadFeed(activeSession: Session | null = session) {
    const client = supabase
    if (!client) {
      return
    }

    setLoadingFeed(true)
    try {
      let loadedPosts: FeedPost[] = []

      const postResult = await client
        .from('posts')
        .select(supportsModelColumns ? MODEL_POST_SELECT : BASIC_POST_SELECT)
        .order('created_at', { ascending: false })

      if (postResult.error && isMissingModelColumnError(postResult.error.message)) {
        setSupportsModelColumns(false)
        const fallbackResult = await client
          .from('posts')
          .select(BASIC_POST_SELECT)
          .order('created_at', { ascending: false })

        if (fallbackResult.error) {
          throw fallbackResult.error
        }
        loadedPosts = (fallbackResult.data ?? []) as unknown as FeedPost[]
      } else if (postResult.error) {
        throw postResult.error
      } else {
        loadedPosts = (postResult.data ?? []) as unknown as FeedPost[]
      }

      setPosts(loadedPosts)

      if (!activeSession) {
        setFeedbackSelections({})
        return
      }

      const reactionResult = await client
        .from('harmful_reactions')
        .select('post_id')
        .eq('user_id', activeSession.user.id)

      if (reactionResult.error) {
        throw reactionResult.error
      }

      const nextSelections = Object.fromEntries(
        ((reactionResult.data ?? []) as HarmfulReactionRow[]).map((entry: HarmfulReactionRow) => [
          entry.post_id,
          'unsafe' as const,
        ]),
      )
      setFeedbackSelections(nextSelections)
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Unable to load the feed.')
    } finally {
      setLoadingFeed(false)
    }
  }

  async function handleAuthSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const client = supabase
    if (!client) {
      return
    }

    setAuthPending(true)
    setStatusMessage(null)

    try {
      if (authMode === 'signup') {
        const { error } = await client.auth.signUp({
          email: email.trim(),
          password,
        })
        if (error) {
          throw error
        }
        setStatusMessage('Account created. If email confirmation is enabled, verify your inbox.')
      } else {
        const { error } = await client.auth.signInWithPassword({
          email: email.trim(),
          password,
        })
        if (error) {
          throw error
        }
      }
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Authentication failed.')
    } finally {
      setAuthPending(false)
    }
  }

  async function handleCreatePost(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const client = supabase
    if (!client || !session) {
      return
    }

    const trimmedCaption = caption.trim()
    if (!trimmedCaption) {
      setStatusMessage('Add a caption before posting.')
      return
    }
    if (!imageFile) {
      setStatusMessage('Choose an image before posting.')
      return
    }

    setSubmitPending(true)
    setStatusMessage(null)
    setModerationVerdict(null)

    let uploadedImagePath: string | null = null
    let verdict: ModerationVerdict | null = null

    try {
      try {
        verdict = await classifyPostBeforePublish(trimmedCaption, imageFile)
        setModerationVerdict(verdict)
      } catch (error) {
        setModerationVerdict(null)
        setStatusMessage(
          error instanceof Error
            ? `${error.message} Posting will continue with model label set to unknown.`
            : 'Moderation failed. Posting will continue with model label set to unknown.',
        )
      }

      uploadedImagePath = createUploadPath(session, imageFile)
      const uploadResult = await client.storage.from(POSTS_BUCKET).upload(uploadedImagePath, imageFile, {
        cacheControl: '3600',
        upsert: false,
        contentType: imageFile.type || 'image/png',
      })
      if (uploadResult.error) {
        throw uploadResult.error
      }

      const legacyInsertPayload = {
        user_id: session.user.id,
        caption: trimmedCaption,
        image_path: uploadedImagePath,
      }

      const modelAwareInsertPayload = {
        ...legacyInsertPayload,
        model_label: toModelLabel(verdict),
        model_ensemble_probability: verdict?.ensembleProbability ?? null,
        model_multimodal_probability: verdict?.multimodalProbability ?? null,
        model_caption_probability: verdict?.captionProbability ?? null,
      }

      let usedLegacyInsert = !supportsModelColumns
      let insertResult

      if (supportsModelColumns) {
        insertResult = await client
          .from('posts')
          .insert(modelAwareInsertPayload)
          .select(MODEL_POST_SELECT)
          .single()

        if (insertResult.error && isMissingModelColumnError(insertResult.error.message)) {
          setSupportsModelColumns(false)
          usedLegacyInsert = true
        }
      }

      if (usedLegacyInsert) {
        insertResult = await client
          .from('posts')
          .insert(legacyInsertPayload)
          .select(BASIC_POST_SELECT)
          .single()
      }

      if (!insertResult) {
        throw new Error('The post could not be saved.')
      }

      if (insertResult.error) {
        throw insertResult.error
      }

      const insertedPost = insertResult.data as FeedPost
      if (usedLegacyInsert && insertedPost?.id) {
        setLocalModelOverrides((current: Record<string, LocalModelOverride>) => ({
          ...current,
          [insertedPost.id]: buildModelOverride(verdict),
        }))
      }

      setCaption('')
      setImageFile(null)
      setModerationVerdict(verdict)
      setIsComposerOpen(false)
      setStatusMessage(
        verdict?.harmful
          ? 'Post published. Seive marked it as potentially harmful.'
          : 'Post published successfully.',
      )
      await loadFeed(session)
    } catch (error) {
      if (uploadedImagePath) {
        await client.storage.from(POSTS_BUCKET).remove([uploadedImagePath])
      }
      setStatusMessage(error instanceof Error ? error.message : 'Unable to publish the post.')
    } finally {
      setSubmitPending(false)
    }
  }

  async function handleDeletePost(post: FeedPost) {
    if (!session?.access_token || post.user_id !== session.user.id) {
      return
    }

    const confirmed = window.confirm('Delete this post? This cannot be undone.')
    if (!confirmed) {
      return
    }

    setDeletingPostId(post.id)
    setStatusMessage(null)
    try {
      await deletePostViaBackend(post.id, session.access_token)
      setPosts((current: FeedPost[]) => current.filter((entry: FeedPost) => entry.id !== post.id))
      setFeedbackSelections((current: Record<string, FeedbackSelection>) => {
        const next = { ...current }
        delete next[post.id]
        return next
      })
      setLocalModelOverrides((current: Record<string, LocalModelOverride>) => {
        const next = { ...current }
        delete next[post.id]
        return next
      })
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Unable to delete the post.')
    } finally {
      setDeletingPostId(null)
    }
  }

  async function handleSetPostSafety(postId: string, isSafe: boolean) {
    const client = supabase
    if (!client || !session) {
      return
    }

    setReactionPendingPostId(postId)
    setStatusMessage(null)
    try {
      if (isSafe) {
        const result = await client
          .from('harmful_reactions')
          .delete()
          .eq('post_id', postId)
          .eq('user_id', session.user.id)

        if (result.error) {
          throw result.error
        }
      } else {
        const result = await client.from('harmful_reactions').upsert(
          {
            post_id: postId,
            user_id: session.user.id,
          },
          {
            onConflict: 'post_id,user_id',
          },
        )

        if (result.error) {
          throw result.error
        }
      }

      setFeedbackSelections((current: Record<string, FeedbackSelection>) => ({
        ...current,
        [postId]: isSafe ? 'safe' : 'unsafe',
      }))
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Unable to save your feedback.')
    } finally {
      setReactionPendingPostId(null)
    }
  }

  async function handleSignOut() {
    const client = supabase
    if (!client) {
      return
    }
    await client.auth.signOut()
  }

  function handleOpenComposer() {
    setStatusMessage(null)
    setIsComposerOpen(true)
  }

  function handleCloseComposer() {
    if (submitPending) {
      return
    }
    setIsComposerOpen(false)
  }

  const signedInUserName = useMemo(() => getDisplayName(session), [session])

  if (!isSupabaseConfigured || !supabase) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          <img className="brand-logo auth-logo" src={seiveLogo} alt="Seive logo" />
          <p className="eyebrow">Setup needed</p>
          <h1>Seive</h1>
          <p className="brand-subtitle">Safest social space just for you</p>
          <p className="empty-copy">
            Add your Supabase environment variables in `frontend/.env` before running the app.
          </p>
          <div className="status-inline">
            <span>Required:</span> `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`
          </div>
        </div>
      </div>
    )
  }

  if (authLoading) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          <img className="brand-logo auth-logo" src={seiveLogo} alt="Seive logo" />
          <h1>Loading Seive...</h1>
        </div>
      </div>
    )
  }

  if (!session) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          <img className="brand-logo auth-logo" src={seiveLogo} alt="Seive logo" />
          <p className="eyebrow">Private community</p>
          <h1>Seive</h1>
          <p className="brand-subtitle">Safest social space just for you</p>
          <form className="auth-form" onSubmit={handleAuthSubmit}>
            <label className="field">
              <span>Email</span>
              <input
                type="email"
                value={email}
                onChange={(event: ChangeEvent<HTMLInputElement>) => setEmail(event.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                required
              />
            </label>
            <label className="field">
              <span>Password</span>
              <input
                type="password"
                value={password}
                onChange={(event: ChangeEvent<HTMLInputElement>) => setPassword(event.target.value)}
                placeholder="Enter your password"
                autoComplete={authMode === 'signin' ? 'current-password' : 'new-password'}
                required
              />
            </label>
            <button className="primary-button auth-button" type="submit" disabled={authPending}>
              {authPending ? 'Working...' : authMode === 'signin' ? 'Sign in' : 'Create account'}
            </button>
          </form>
          <button
            className="secondary-button toggle-auth-button"
            type="button"
            onClick={() =>
              setAuthMode((current: 'signin' | 'signup') =>
                current === 'signin' ? 'signup' : 'signin',
              )
            }
          >
            {authMode === 'signin' ? 'Need an account? Sign up' : 'Already have an account? Sign in'}
          </button>
          {statusMessage ? <p className="status-copy">{statusMessage}</p> : null}
        </div>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <img className="brand-logo" src={seiveLogo} alt="Seive logo" />
          <div>
            <div className="brand-title-row">
              <h1>Seive</h1>
              <span className="user-chip">{signedInUserName}</span>
            </div>
            <p className="brand-subtitle">Safest social space just for you</p>
          </div>
        </div>

        <div className="topbar-actions">
          <button
            className="action-button create-post-button"
            type="button"
            onClick={handleOpenComposer}
          >
            <span className="button-icon">＋</span>
            <span className="button-label">Create Post</span>
          </button>
          <button className="action-button signout-button" type="button" onClick={handleSignOut}>
            <span className="button-icon">⇦</span>
            <span className="button-label">Sign Out</span>
          </button>
          <div className="avatar-badge" aria-hidden="true">
            {getAvatarInitial(session)}
          </div>
        </div>
      </header>

      <main className="content-grid content-grid-single">
        <section className="feed-column">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Community feed</p>
              <h2>Recent posts</h2>
            </div>
            <button className="secondary-button refresh-button" type="button" onClick={() => void loadFeed(session)}>
              Refresh
            </button>
          </div>

          {loadingFeed ? <div className="empty-state">Loading posts...</div> : null}

          {!loadingFeed && posts.length === 0 ? (
            <div className="empty-state">
              <p className="empty-copy">No posts yet. Create the first one to start the feed.</p>
              <button className="primary-button empty-state-button" type="button" onClick={handleOpenComposer}>
                Create Post
              </button>
            </div>
          ) : null}

          <div className="feed-list">
            {posts.map((post) => {
              const modelState = getModelState(post, localModelOverrides)
              const postImageUrl = getImageUrl(post.image_path)
              const selection = feedbackSelections[post.id]
              const isOwner = post.user_id === session.user.id
              const isHarmful = modelState.modelLabel === 'harmful'

              return (
                <article
                  key={post.id}
                  className={`post-card ${isHarmful ? 'post-card-harmful' : ''}`}
                >
                  <div className="post-header">
                    <div>
                      <p className="post-meta">{formatPostTime(post.created_at)}</p>
                      <h3>{post.caption}</h3>
                    </div>
                    {isOwner ? (
                      <button
                        className="delete-post-button"
                        type="button"
                        onClick={() => void handleDeletePost(post)}
                        disabled={deletingPostId === post.id}
                        aria-label="Delete post"
                      >
                        {deletingPostId === post.id ? '…' : '🗑'}
                      </button>
                    ) : null}
                  </div>

                  {isHarmful ? (
                    <div className="model-alert">
                      Seive thinks that this post contains harmful content.
                    </div>
                  ) : null}

                  {postImageUrl ? (
                    <div className="post-image-frame">
                      <img src={postImageUrl} alt={post.caption} />
                    </div>
                  ) : null}

                  <div className="post-footer">
                    <div className="model-chip-row">
                      <span className={`model-chip ${modelState.modelLabel}`}>
                        Model: {modelState.modelLabel}
                      </span>
                      {modelState.modelEnsembleProbability !== null ? (
                        <span className="probability-chip">
                          Score {modelState.modelEnsembleProbability.toFixed(3)}
                        </span>
                      ) : null}
                    </div>

                    <div className="feedback-row">
                      <span className="feedback-question">Do you find this safe?</span>
                      <div className="vote-actions">
                        <button
                          className={`vote-button safe ${selection === 'safe' ? 'active' : ''}`}
                          type="button"
                          onClick={() => void handleSetPostSafety(post.id, true)}
                          disabled={reactionPendingPostId === post.id}
                          aria-label="Mark post as safe"
                        >
                          👍
                        </button>
                        <button
                          className={`vote-button unsafe ${selection === 'unsafe' ? 'active' : ''}`}
                          type="button"
                          onClick={() => void handleSetPostSafety(post.id, false)}
                          disabled={reactionPendingPostId === post.id}
                          aria-label="Mark post as unsafe"
                        >
                          👎
                        </button>
                      </div>
                    </div>
                  </div>
                </article>
              )
            })}
          </div>
        </section>
      </main>

      {isComposerOpen ? (
        <div className="modal-overlay" role="presentation" onClick={handleCloseComposer}>
          <section
            className="composer-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="create-post-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="composer-modal-header">
              <div>
                <p className="eyebrow">Share something</p>
                <h2 id="create-post-title">Create a post</h2>
              </div>
              <button
                className="modal-close-button"
                type="button"
                onClick={handleCloseComposer}
                disabled={submitPending}
                aria-label="Close create post modal"
              >
                ×
              </button>
            </div>

            <form className="composer-form" onSubmit={handleCreatePost}>
              <label className="field">
                <span>Caption</span>
                <textarea
                  value={caption}
                  onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
                    setCaption(event.target.value)
                  }
                  placeholder="What would you like to share today?"
                  rows={4}
                  required
                />
              </label>

              <label className="upload-field" htmlFor="post-image">
                <span className="upload-icon">🖼</span>
                <span>{imageFile ? imageFile.name : 'Choose an image'}</span>
              </label>
              <input
                id="post-image"
                className="visually-hidden"
                type="file"
                accept="image/*"
                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                  setImageFile(event.target.files?.[0] ?? null)
                }
              />

              {imagePreviewUrl ? (
                <div className="image-preview-frame">
                  <img src={imagePreviewUrl} alt="Selected preview" />
                </div>
              ) : null}

              {moderationVerdict ? (
                <div className={`moderation-preview ${moderationVerdict.harmful ? 'harmful' : 'safe'}`}>
                  <strong>{moderationVerdict.harmful ? 'Potentially harmful' : 'Looks safe'}</strong>
                  <span>
                    Ensemble score {moderationVerdict.ensembleProbability.toFixed(3)} against threshold{' '}
                    {moderationVerdict.ensembleThreshold.toFixed(2)}
                  </span>
                </div>
              ) : null}

              {statusMessage ? <p className="status-copy">{statusMessage}</p> : null}

              <div className="composer-modal-actions">
                <button
                  className="secondary-button modal-secondary-button"
                  type="button"
                  onClick={handleCloseComposer}
                  disabled={submitPending}
                >
                  Cancel
                </button>
                <button className="primary-button submit-button" type="submit" disabled={submitPending}>
                  {submitPending ? 'Publishing...' : 'Publish Post'}
                </button>
              </div>
            </form>
          </section>
        </div>
      ) : null}
    </div>
  )
}

export default App
