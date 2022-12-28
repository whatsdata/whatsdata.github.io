---
title: "Post by year"
layout: posts
permalink: /year/
author_profile: true
sidebar_main: true
---

{% assign postsByYear = site.tags.stat| group_by_exp: 'post', 'post.date | date: "%Y"' %}
