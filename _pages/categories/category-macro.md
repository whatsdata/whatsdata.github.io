---
title: "macro"
layout: archive
permalink: categories/macro
author_profile: true
sidebar_main_econ: true
---

{% assign posts = site.econs.macro %}
{% for post in posts %} {% include archive-single2.html type=page.entries_layout %} {% endfor %}
