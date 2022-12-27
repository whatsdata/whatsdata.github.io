---
title: "Economics"
layout: archive
permalink: econ
author_profile: true
sidebar_main: false
sidebar_main_econ: true
---

{% assign posts = site.categories.econ %}
{% for post in posts %} {% include archive-single2.html type=page.entries_layout %} {% endfor %}
