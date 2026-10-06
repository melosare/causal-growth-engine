# Causal Growth Engine — Causal Assumptions

## 1. Purpose

The Causal Growth Engine (CGE) estimates the causal effect of controllable
growth interventions on organic growth and downstream revenue.

The initial system is designed to answer questions such as:

- What causes changes in organic installs?
- How much incremental organic growth is attributable to each intervention?
- How does incremental organic growth translate into revenue?
- What is the expected revenue impact of changing the growth strategy?
- How should a finite budget be allocated across controllable interventions?

The CGE is intended for decision support and counterfactual analysis. It does
not assume that observational relationships are causal without explicit
identification assumptions.

---

## 2. Unit of Analysis

The initial modeling unit is:

    product × day

Each observation represents one product/title on one calendar day.

The initial synthetic dataset will contain multiple products observed over
multiple days so that the model can represent:

- product-level heterogeneity
- temporal variation
- seasonality
- product maturity
- correlated interventions
- repeated observations within products

The initial target scale is approximately:

    30 products × 365 days = 10,950 observations

The architecture must not assume that this scale is fixed.

---

## 3. Variables

### 3.1 Controllable interventions

These are variables the growth/product organization can intentionally change
and which the optimizer may eventually control.

#### Paid UA spend

    paid_ua_spend

Monetary investment in paid user acquisition.

Expected role:

    intervention

Potential causal pathways include:

    paid_ua_spend
        ↓
    paid_installs
        ↓
    social activity / product visibility
        ↓
    organic_installs

Paid UA may also have a direct effect on organic growth through increased
awareness, ranking, social proof, or network effects.

---

#### Influencer spend

    influencer_spend

Investment in influencer/creator activity.

Expected role:

    intervention

Potential pathways include:

    influencer_spend
        ↓
    social exposure
        ↓
    social engagement
        ↓
    organic_installs

Influencer activity may also have a direct effect on organic acquisition.

---

#### Social media posts

    social_media_posts

Number or intensity of organization-controlled social media posts.

Expected role:

    intervention

Potential pathways include:

    social_media_posts
        ↓
    social_media_likes
    social_media_comments
        ↓
    organic_installs

Post volume should not automatically be interpreted as causal without
accounting for content quality, audience demand, timing, and product state.

---

#### Product test release

    product_test_release

Indicator or intensity of a product test release.

Expected role:

    intervention

Potential pathways include:

    product_test_release
        ↓
    product experience / quality
        ↓
    engagement / rating
        ↓
    organic_installs
        ↓
    revenue

The initial model will represent this as an intervention. The exact treatment
encoding will be determined by the data available in the real system.

---

#### Product version update

    product_version_update

Indicator or intensity of a production product version update.

Expected role:

    intervention

Potential pathways include:

    product_version_update
        ↓
    product experience / quality
        ↓
    app_store_rating
    engagement
    retention
        ↓
    organic_installs / revenue

Updates may have positive or negative effects depending on their quality.

---

## 4. Mediators and Mechanism Variables

The following variables are important to the causal system but should not
initially be treated as independently controllable optimization variables.

### Paid installs

    paid_installs

Expected role:

    mediator / downstream consequence of paid UA

Primary relationship:

    paid_ua_spend
        ↓
    paid_installs

Paid installs may subsequently affect organic growth through awareness,
social activity, ranking, network effects, or other mechanisms.

Therefore, conditioning on paid_installs when estimating the total causal
effect of paid_ua_spend can block part of the treatment effect.

The CGE must distinguish:

    total effect of paid_ua_spend

from:

    direct effect of paid_ua_spend

and:

    mediated effect through paid_installs

where the available data and identification assumptions support such
decomposition.

---

### Social media likes

    social_media_likes

Expected role:

    mediator / behavioral outcome

Likes may be influenced by:

- social media posts
- influencer activity
- product popularity
- underlying demand
- product quality
- external events

Likes should not initially be treated as an independent intervention.

---

### Social media comments

    social_media_comments

Expected role:

    mediator / behavioral outcome

Comments may be influenced by:

- social media posts
- influencer activity
- product popularity
- underlying demand
- product quality
- external events

Comments should not initially be treated as an independent intervention.

---

### App store rating

    app_store_rating

Expected role:

    mediator / product-state variable

Potential pathway:

    product_test_release
            ↓
    product_version_update
            ↓
      product quality
            ↓
     app_store_rating
            ↓
      organic_installs

Rating is affected by historical user behavior and product experience.
Therefore, it should not initially be treated as a freely controllable
treatment.

---

## 5. Primary Outcome

### Organic installs

    organic_installs

This is the primary causal outcome for V1.

The first major modeling question is:

    What causes changes in organic_installs?

The CGE should estimate the expected change in organic installs under
counterfactual interventions.

For example:

    E[Y | do(paid_ua_spend = x + Δ)]

versus:

    E[Y | do(paid_ua_spend = x)]

where Y represents organic installs.

---

## 6. Downstream Business Outcomes

### Engagement

    engagement

Engagement represents user activity following acquisition.

It is expected to be influenced by:

- product quality
- product updates
- organic acquisition
- paid acquisition
- social activity
- user composition

Engagement may act as a mediator between acquisition and revenue.

---

### Retention

    retention

Retention represents continued user activity after acquisition.

It is expected to be influenced by:

- product quality
- product updates
- user composition
- engagement
- acquisition source

Retention may act as a mediator between acquisition and long-term revenue.

---

### Revenue

    revenue

Revenue is the primary downstream business outcome.

The initial conceptual structure is:

    interventions
          ↓
    organic_installs
          ↓
    engagement
          ↓
    retention
          ↓
       revenue

The actual model may contain additional direct pathways into revenue.

The CGE should ultimately quantify:

    incremental revenue

associated with counterfactual changes in controllable interventions.

---

## 7. Initial Causal DAG

The initial causal hypothesis is:

```text
                           Market / Demand
                          /       |       \
                         /        |        \
                        ▼         ▼         ▼
                 Marketing     Product    Organic
                 Decisions      State     Demand
                    │             │          │
        ┌───────────┼─────────────┤          │
        │           │             │          │
        ▼           ▼             ▼          │
    Paid UA     Influencer    Product        │
     Spend        Spend        Actions       │
        │           │          /    \         │
        │           │         ▼      ▼        │
        │           │      Test     Version   │
        │           │     Release   Update    │
        │           │         │       │       │
        ▼           ▼         └──┬────┘       │
   Paid Installs   Social         │            │
        │          Activity       │            │
        │          /    \          │            │
        │         ▼      ▼         ▼            │
        │       Likes  Comments  Rating         │
        │         \      |        /             │
        │          \     |       /              │
        └───────────┴─────┴─────┘               │
                         │                      │
                         └──────────┬───────────┘
                                    ▼
                              Organic Installs
                                    │
                              ┌─────┴─────┐
                              ▼           ▼
                         Engagement    Retention
                              │           │
                              └─────┬─────┘
                                    ▼
                                  Revenue
