# Venues to Add: High-Profile Ingestion Strategy

A comprehensive, prioritized action plan for targeting high-profile New York City cultural, comedy, and performance venues that currently show **0 events** in the database.

## 📊 Executive Summary
* **Total Venues in Cache**: 5637
* **Total Venues with 0 Events**: 3711 (65.8% of cache)
* **High-Profile Venues with 0 Events**: 1328
* **Eventbrite Usage Rate**: ~6.0% of top venues (est. **111 total venues** in our zero-event cache)

---

## 🎯 1. High-Profile Custom Scrapers (Top Priority)
The following cultural landmarks and tour organizations are extremely high-profile targets. Standard API-based discovery (like Ticketmaster) does not cover them adequately, so they require **dedicated, custom scraping solutions**.

### 🏛️ The Metropolitan Museum of Art (The Met)
* **Status**: High profile (has some events under "The Met" but 0 events for "The Metropolitan Museum of Art" and "The Met Cloisters").
* **Official URL**: [metmuseum.org/exhibitions](https://www.metmuseum.org/exhibitions) and [metmuseum.org/events/programs](https://www.metmuseum.org/events/programs)
* **Scraping Strategy**: Parse the Met's public exhibition and event calendar web pages. A custom BeautifulSoup/Playwright scraper should crawl their program pages, extracting exhibition titles, start/end dates, program descriptions, and link coordinates.

### 🖼️ Museum of Modern Art (MoMA)
* **Status**: High profile (0 events in Firestore, except MoMA PS1 which has 3).
* **Official URL**: [moma.org/calendar](https://www.moma.org/calendar)
* **Scraping Strategy**: Parse MoMA's unified calendar REST endpoints or crawl `/calendar/events` using selectors. MoMA loads calendar data via a client-side API call that returns JSON; capturing this network payload directly avoids HTML parsing overhead.

### 🎙️ Comedy Cellar (and associated venues)
* **Status**: High profile (requires own custom solution). Includes *Village Underground* and *Fat Black Pussycat*.
* **Official URL**: [comedycellar.com/line-up/](https://www.comedycellar.com/line-up/)
* **Scraping Strategy**: Scrape the unified reservations calendar on the Comedy Cellar website. The schedules and comedian line-ups for all rooms (Main Room, Village Underground, Fat Black Pussycat) are loaded dynamically. We need a custom parser to extract the showtime slots, comedian lists, and reservation links.

### 🎶 92nd Street Y (92NY)
* **Status**: High profile (9-11 events, but incomplete / needs robust daily scraper).
* **Official URL**: [92ny.org/events](https://www.92ny.org/events)
* **Scraping Strategy**: Crawl their events listings page using daily date-based pagination, or check for RSS feeds/JSON APIs on their event search endpoints.

### 🥾 Municipal Art Society (MAS) Tours [NEW TARGET]
* **Status**: *Not currently in database*. High priority addition.
* **Official URL**: [mas.org/tours/](https://www.mas.org/tours/)
* **Scraping Strategy**: Crawl the MAS tours catalog to ingest educational neighborhood walks, architectural tours, and historic preservation events into our system.

### 🏙️ Open House New York (Open NY) Tours [NEW TARGET]
* **Status**: *Not currently in database*. High priority addition.
* **Official URL**: [ohny.org](https://ohny.org)
* **Scraping Strategy**: Scrape the official OHNY calendar, tracking year-round walking tours, special access events, and their major annual October weekend festival.

---

## ⚡ 2. Eventbrite Integration Study & API Limits
The user requested an audit of remaining venues to see how many use Eventbrite and a review of Eventbrite's developer capabilities.

### 🔑 Eventbrite Developer API Limits
* **Hourly Request Limit**: **2,000 calls per hour** (per OAuth token).
* **Daily Request Limit**: **48,000 calls per day**.
* **Failure Response**: Returns `429 HIT_RATE_LIMIT` when exceeded.

> [!IMPORTANT]
> **The Late-2019 Deprecation Trap**:
> Eventbrite permanently disabled their generic public `event_search` endpoint in late 2019. Consequently, **it is impossible to search for public events by city/category** using their standard API.
> To query Eventbrite for events, we must have the explicit **Eventbrite Venue ID** or **Eventbrite Organizer (Organization) ID** to call endpoints such as:
> * `GET /v3/venues/{venue_id}/events/`
> * `GET /v3/organizations/{organization_id}/events/`

### 📊 Empirical Venue Scan Results
We ran a concurrent, live inspection on the homepages of the **top 100 zero-event venues** in our database:
* **Eventbrite Usage Rate**: **6.0%** of high-profile zero-event venues explicitly use Eventbrite for event listings or ticketing links.
* **Extrapolated Impact**: Across our 1,857 zero-event venues that have registered websites, **approximately 111 venues** use Eventbrite.
* **Identified Eventbrite Venues**:
  1. *Club Groove* (https://www.clubgroovenyc.com/music-calendar)
  2. *Producers Club Theaters* (https://producersclub.com/shows/)
  3. *Manhattan Center* (https://www.mc34.com/)
  4. *Brooklyn Navy Yard Center at BLDG 92* (http://bldg92.org/events)
  5. *Alice Austen House Museum* (https://aliceausten.org/events/exhibition-opening-resilient-communities/)
  6. *Aaron Davis Hall at The City College of New York* (https://citycollegecenterforthearts.org/events)

#### 🛠️ Recommended Eventbrite Scraping Workflow
1. **Organizer Profile Discovery**: Automate a crawl of the venue homepages to find links containing `eventbrite.com/o/` or `eventbrite.com/e/`.
2. **ID Extraction**: Parse the Organizer ID from the profile URL.
3. **API Fetching**: Query the official Eventbrite API using the extracted Organizer ID to reliably ingest clean, structured event lists without needing full HTML scrapers.

---

## 🔝 3. Top 50 Targeted Zero-Event Venues
These are the top 50 venues currently showing **0 events** in our Firestore database. They are ranked by an importance score combining category classification weight, presence of a website, and name keywords (excluding our 6 special targets above).

| Rank | Score | Venue Name | Category | Neighborhood | Website / Events URL |
|---|---|---|---|---|---|
| 1 | **38** | **Jazz at Lincoln Center: Rose Theater** | classical music venues | Midtown | [Events Link](https://jazz.org/watch-listen-discover/program-archive/) |
| 2 | **35** | **Greenpoint Comedy Club** | comedy clubs standup | Greenpoint | [Website](https://greenpointcomedy.com/) |
| 3 | **35** | **Haugen Hall Theatre at Staten Island Academy** | concert halls | Todt Hill | [Events Link](https://www.statenislandacademy.org/about/event-calendar) |
| 4 | **35** | **Morgan Library & Museum - Gilder Lehrman Hall** | concert halls | Murray Hill | [Events Link](https://www.themorgan.org/drawing-institute/events) |
| 5 | **35** | **Smoke Jazz Club** | live music venues | Morningside Heights | [Events Link](https://smokejazz.com/) |
| 6 | **34** | **New Stage Theatre Company** | opera houses | Midtown | [Website](https://www.newstagetheatre.org/) |
| 7 | **34** | **Primary Stages at 59E59 Theaters** | off-broadway theaters | Midtown East | [Website](https://59e59.org/) |
| 8 | **33** | **Alvin Ailey American Dance Theater at the Joan Weill Center for Dance** | dance venues ballet contemporary | Midtown | [Website](https://www.alvinailey.org/) |
| 9 | **33** | **Elebash Recital Hall - CUNY Graduate Center** | classical music venues | Midtown | [Events Link](https://www.gc.cuny.edu/about-graduate-center/facilities-and-services/room-reservations/elebash-recital-hall/events.html) |
| 10 | **33** | **Jazz at Lincoln Center** | best live music venues | Columbus Circle | [Website](https://jazz.org/) |
| 11 | **33** | **The Jazz Gallery** | jazz clubs | Flatiron | [Website](https://www.jazzgallery.org/) |
| 12 | **33** | **Metropolitan Opera House (Lincoln Center)** | opera houses | Upper West Side | [Website](https://www.metopera.org/) |
| 13 | **33** | **National Opera Center** | opera houses | Midtown | [Website](https://www.operaamerica.org/national-opera-center) |
| 14 | **33** | **OPERA America's National Opera Center** | classical music venues | Chelsea | [Website](https://www.operaamerica.org/national-opera-center) |
| 15 | **33** | **Signature Theatre - The Pershing Square Signature Center** | opera houses | Hell's Kitchen | [Website](https://www.signaturetheatre.org/) |
| 16 | **30** | **Aaron Davis Hall** | concert halls | Harlem | [Events Link](https://citycollegecenterforthearts.org/) |
| 17 | **30** | **Aaron Davis Hall at The City College of New York** | opera houses | Harlem | [Events Link](https://citycollegecenterforthearts.org/events) |
| 18 | **30** | **Alvin Ailey American Dance Theater** | dance venues ballet contemporary | Hell's Kitchen | [Website](https://www.alvinailey.org/) |
| 19 | **30** | **Baisley Powell Elebash Recital Hall** | classical music venues | Greenwich Village | [Events Link](https://www.gc.cuny.edu/about-graduate-center/facilities-and-services/room-reservations/elebash-recital-hall/events.html) |
| 20 | **30** | **Bernard Museum of Judaica** | art museums | Upper East Side | [Website](https://www.emanuelnyc.org/about-us/bernard-museum/) |
| 21 | **30** | **BKLYN Comedy Club** | comedy clubs standup | Bushwick | [Website](https://www.bklyncomedyclub.com/) |
| 22 | **30** | **Bone Museum** | art museums | Williamsburg | [Website](https://www.thebonemuseum.com/) |
| 23 | **30** | **The Bronx Beer Hall** | concert halls | Belmont | [Events Link](https://thebronxbeerhall.com/events) |
| 24 | **30** | **Bronx Opera Company** | opera houses | Norwood | [Website](https://www.bronxopera.org/) |
| 25 | **30** | **Brooklyn Museum** | art museums | Crown Heights | [Website](https://www.brooklynmuseum.org/) |
| 26 | **30** | **Bushwick Comedy Club** | improv comedy theaters | Bushwick | [Website](https://www.bushwickcomedy.com/) |
| 27 | **30** | **The Cell Theatre** | classical music venues | Chelsea | [Website](https://www.thecelltheatre.org/) |
| 28 | **30** | **Cellar Dog** | jazz clubs | West Village | [Website](https://www.cellardog.net/) |
| 29 | **30** | **Club Groove** | concert halls | Greenwich Village | [Events Link](https://www.clubgroovenyc.com/music-calendar) |
| 30 | **30** | **Cobra Club** | concert halls | Bushwick | [Events Link](http://cobraclubbk.com/shows-and-events) |
| 31 | **30** | **Dizzy's Club Coca-Cola** | best live music venues | Upper West Side | [Website](https://jazz.org/dizzys/) |
| 32 | **30** | **DR2 Theatre** | off-broadway theaters | Union Square | [Website](https://www.darerorun.org/) |
| 33 | **30** | **Elebash Recital Hall** | classical music venues | Midtown | [Events Link](https://www.gc.cuny.edu/about-graduate-center/facilities-and-services/room-reservations/elebash-recital-hall/events.html) |
| 34 | **30** | **The Flea Theater** | off-broadway theaters | TriBeCa | [Website](https://theflea.org/) |
| 35 | **30** | **Florence Gould Hall at FIAF** | classical music venues | Midtown East | [Website](https://fiaf.org/) |
| 36 | **30** | **Florence Gould Hall at French Institute Alliance Française (FIAF)** | classical music venues | Midtown East | [Website](https://fiaf.org/) |
| 37 | **30** | **Gotham Comedy Club** | comedy clubs standup | Chelsea | [Website](https://gothamcomedyclub.com/) |
| 38 | **30** | **Gottscheer Hall** | live music venues | Ridgewood | [Events Link](http://gottscheerhall.com/events) |
| 39 | **30** | **Ice Theatre of New York** | dance venues ballet contemporary | NYC | [Website](https://www.icetheatre.org/) |
| 40 | **30** | **The Juilliard School - Morse Hall** | concert halls | Upper West Side | [Events Link](https://www.juilliard.edu/events.html) |
| 41 | **30** | **The Juilliard School - Morse Recital Hall** | opera houses | Lincoln Square | [Events Link](https://www.juilliard.edu/events.html) |
| 42 | **30** | **The Juilliard School - Paul Recital Hall** | concert halls | Upper West Side | [Events Link](https://www.juilliard.edu/campus-life/performance-facilities/paul-recital-hall/events.html) |
| 43 | **30** | **The Juilliard School - Peter Jay Sharp Theater** | concert halls | Upper West Side | [Events Link](https://www.juilliard.edu/campus-life/performance-venues/peter-jay-sharp-theater/events.html) |
| 44 | **30** | **Mannes School of Music - Mannes Concert Hall** | classical music venues | Greenwich Village | [Website](https://www.newschool.edu/mannes/) |
| 45 | **30** | **Metropolitan Opera** | opera houses | Upper West Side | [Website](https://www.metopera.org/) |
| 46 | **30** | **Metropolitan Opera House** | concert halls | Upper West Side | [Events Link](https://www.metopera.org/season/events/) |
| 47 | **30** | **The Museum of Broadway** | art museums | Times Square/Theatre District | [Website](https://www.themuseumofbroadway.com/) |
| 48 | **30** | **Museum of Ice Cream** | museums | SoHo | [Website](https://www.museumoficecream.com/new-york-city) |
| 49 | **30** | **Museum of Sex** | museums | Nomad | [Website](https://www.museumofsex.com/) |
| 50 | **30** | **National Museum of the American Indian** | art museums | Lower Manhattan | [Website](https://americanindian.si.edu/) |

---

## 🏛️ 4. Original Category Lists (Reference)
These are all high-profile zero-event venues in the database grouped by historical category for reference.


### lecture halls universities (104 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Academic Core Building Lecture Halls** | Jamaica | `None` | `None` | [Events Link](https://www.york.cuny.edu/administrative/office-of-facilities-and-planning/special-events) |
| **Amphitheater, York College** | Jamaica | `None` | `None` | None |
| **Anna-Maria and Stephen Kellen Auditorium, Sheila C. Johnson Design Center, The New School** | Greenwich Village | `None` | `None` | None |
| **Baruch College - Bernie West Theatre** | Gramercy Park | `None` | `None` | None |
| **Boylan Hall Lecture Hall, Brooklyn College, CUNY** | Midwood | `None` | `None` | [Events Link](https://www.brooklyn.edu/news-events/) |
| **Brooklyn Law School - Conference Center** | Downtown Brooklyn | `None` | `None` | [Events Link](https://www.brooklaw.edu/news-and-events/events/2026/2026_02_24/) |
| **CUNY Graduate Center - Elebash Recital Hall** | Midtown | `None` | `None` | [Events Link](https://www.gc.cuny.edu/about-graduate-center/facilities-and-services/room-reservations/elebash-recital-hall/events.html) |
| **CUNY Graduate Center - Proshansky Auditorium** | Midtown | `None` | `None` | [Events Link](https://www.gc.cuny.edu/about-graduate-center/facilities-and-room-rentals/proshansky-auditorium/events.html) |
| **Campus Center Green Dolphin Lounge** | Willowbrook | `None` | `None` | None |
| **Center for the Arts Lecture Hall** | Willowbrook | `None` | `None` | None |
| **Claire Tow Theater, Brooklyn College** | Midwood | `None` | `None` | [Events Link](https://brooklyn.edu/tow/events.html) |
| **Columbia University - Schapiro Hall Auditorium** | Morningside Heights | `None` | `None` | [Events Link](https://eventmanagement.columbia.edu/content/schapiro-hall-auditorium/events.html) |
| **Cornell University ILR School - Conference Center** | Midtown East | `None` | `None` | None |
| **D'Angelo Center Ballroom, St. John's University** | Jamaica | `None` | `None` | None |
| **Doremus Lecture Hall, Baskerville Hall, The City College of New York** | Hamilton Heights | `None` | `None` | [Events Link](https://www.ccny.cuny.edu/calendar/milt-stern-workshop-pathways-careers-financial-advising-3) |
| **East Academic Complex Lecture Halls** | Mott Haven | `None` | `None` | None |
| **Eisner & Lubin Auditorium, Kimmel Center for University Life, NYU** | Greenwich Village | `None` | `None` | [Website](https://www.nyu.edu/students/student-information-and-resources/student-community-outreach/kimmel-center/spaces/eisner-lubin-auditorium.html) |
| **Fashion Institute of Technology - Haft Theater** | Chelsea | `None` | `None` | None |
| **Fashion Institute of Technology - Katie Murphy Amphitheatre** | Chelsea | `None` | `None` | None |
| **Fisher Center** | NoMad | `None` | `None` | [Events Link](https://nyc.syr.edu/events/) |
| **Fordham Law School - Skadden Conference Center** | Lincoln Square | `None` | `None` | None |
| **Fordham University School of Law - Skadden Conference Center** | Lincoln Square | `None` | `None` | None |
| **Founders Hall Auditorium** | Brooklyn Heights | `None` | `None` | None |
| **Frederick Loewe Theatre, NYU** | Greenwich Village | `None` | `None` | None |
| **Havemeyer Hall 309** | Morningside Heights | `None` | `None` | None |
| **Havemeyer Hall, Room 309** | Morningside Heights | `None` | `None` | [Events Link](https://www.columbia.edu/events.html) |
| **Higgins Hall Auditorium** | Clinton Hill | `None` | `None` | None |
| **Hostos Community College Main Theater** | Mott Haven | `None` | `None` | [Website](https://www.hostoscenter.org/) |
| **Hostos Community College Repertory Theater** | Mott Haven | `None` | `None` | [Website](https://www.hostoscenter.org/) |
| **Humanities Building Lecture Hall** | Bayside | `None` | `None` | None |
| **Hunter College - Ida K. Lang Recital Hall** | Upper East Side | `None` | `None` | None |
| **Hunter College - Lecture Halls** | Upper East Side | `None` | `None` | None |
| **Ingersoll Hall Lecture Hall, Brooklyn College, CUNY** | Midwood | `None` | `None` | [Events Link](https://www.brooklyn.edu/wp-json/tribe/events/v1/) |
| **John A. Paulson Center, NYU** | Greenwich Village | `None` | `None` | [Website](https://www.nyu.edu/about/university-initiatives/paulson-center.html) |
| **King Juan Carlos I of Spain Center Auditorium** | Greenwich Village | `None` | `None` | [Events Link](https://www.kjcc.org/events/) |
| **Kumble Theater for the Performing Arts** | Downtown Brooklyn | `None` | `None` | None |
| **Lecture Hall, 370 Jay Street, NYU** | Downtown Brooklyn | `None` | `None` | None |
| **Lecture Hall, Center for the Arts, College of Staten Island** | Willowbrook | `None` | `None` | None |
| **Lehman College - East Dining Hall** | Bedford Park | `None` | `None` | None |
| **Lehman College - Faculty Dining Hall** | Bedford Park | `None` | `None` | None |
| **Lehman College - Studio Theatre** | Bedford Park | `None` | `None` | None |
| **Leonard & Claire Tow Center for the Performing Arts** | Midwood | `None` | `None` | [Events Link](https://brooklyn.edu/tow-center/events.html) |
| **Little Theater** | Long Island City | `None` | `None` | None |
| **Little Theater, St. John's University** | Jamaica | `None` | `None` | None |
| **Lovinger Theatre, Lehman College** | Bedford Park | `None` | `None` | None |
| **Main Hall Theatre** | Grymes Hill | `None` | `None` | [Events Link](https://www.lpac.nyc/upcoming-events/rdf-2026) |
| **Marillac Hall Auditorium, St. John's University** | Jamaica | `None` | `None` | None |
| **Maritime Academic Center Multi-Purpose Room** | Throgs Neck | `None` | `None` | None |
| **Maritime Academic Center Multi-Purpose Room, SUNY Maritime College** | Throggs Neck | `None` | `None` | None |
| **Mary Pickett Lecture Hall, Medgar Evers College** | Crown Heights | `None` | `None` | [Events Link](https://www.mec.cuny.edu/events/?ical=1) |
| **Memorial Hall Auditorium, Pratt Institute** | Clinton Hill | `None` | `None` | None |
| **Memorial Hall, Pratt Institute** | Clinton Hill | `None` | `None` | None |
| **Myrtle Hall Lecture Rooms** | Clinton Hill | `None` | `None` | None |
| **NYC Seminar & Conference Center - Event Hall 2** | NoMad | `None` | `None` | [Website](https://nycseminars.com/) |
| **NYC Seminar & Conference Center - Meeting Room 1** | NoMad | `None` | `None` | [Website](https://nycseminarcenter.com/) |
| **NYC Seminar & Conference Center - Seminar Room A** | NoMad | `None` | `None` | [Website](https://nycseminarcenter.com/) |
| **NYC Seminar & Conference Center - Seminar Room C** | NoMad | `None` | `None` | [Website](https://www.nycseminarcenter.com/) |
| **NYU Global Center for Academic and Spiritual Life - Grand Hall** | Greenwich Village | `None` | `None` | [Website](https://www.nyu.edu/life/resources-and-services/nyu-event-spaces/global-center-for-academic-and-spiritual-life.html) |
| **NYU Grossman School of Medicine - Alumni Hall B (Berman Lecture Hall)** | Kips Bay | `None` | `None` | [Events Link](https://med.nyu.edu/calendar/) |
| **NYU Kimmel Center for University Life - Eisner & Lubin Auditorium** | Greenwich Village | `None` | `None` | [Website](https://www.nyu.edu/students/communities-and-groups/student-centers/kimmel/spaces/eisner-lubin-auditorium.html) |
| **NYU Kimmel Center for University Life - Rosenthal Pavilion** | Greenwich Village | `None` | `None` | [Website](https://www.nyu.edu/students/communities-and-groups/student-centers/kimmel.html) |
| **NYU Langone Medical Center Alumni Hall** | Kips Bay | `None` | `None` | None |
| **NYU School of Law - Vanderbilt Hall** | Greenwich Village | `None` | `None` | None |
| **NYU Silver Center for Arts and Science - Hemmerdinger Hall** | Greenwich Village | `None` | `None` | [Website](https://www.nyu.edu/life/resources-and-services/nyu-event-spaces/silver-center.html) |
| **NYU Wasserman Center - Presentation Room B** | East Village | `None` | `None` | [Website](https://www.nyu.edu/students/communities-and-groups/student-success/wasserman-center.html) |
| **New York College of Podiatric Medicine - Lecture Hall** | East Harlem | `None` | `None` | None |
| **New York Law School - Events Center** | Tribeca | `None` | `None` | [Events Link](https://www.nyls.edu/events/) |
| **New York University - Global Center for Academic and Spiritual Life - Medium Spaces** | Greenwich Village | `None` | `None` | [Website](https://www.nyu.edu/students/communities-and-groups/student-community-center/global-center-for-academic-and-spiritual-life/medium-spaces.html) |
| **New York University - Kimmel Center for University Life - Medium Spaces** | Greenwich Village | `None` | `None` | [Website](https://www.nyu.edu/students/communities-and-groups/student-centers/kimmel.html) |
| **New York University School of Law - Lipton Hall** | Greenwich Village | `None` | `None` | None |
| **Pace University - Michael Schimmel Center for the Arts** | Financial District | `None` | `None` | [Website](https://schimmelcenter.org/) |
| **Pace University - Schimmel Center for the Arts** | Financial District | `None` | `None` | [Website](https://schimmelcenter.org/) |
| **Pace University Pleasantville - Kessel Student Center, Butcher Suite** | Pleasantville | `None` | `None` | None |
| **Pope Auditorium, Fordham University at Lincoln Center** | Lincoln Square | `None` | `None` | None |
| **Pratt Institute - Higgins Hall Auditorium** | Clinton Hill | `None` | `None` | None |
| **Pratt Institute - Memorial Hall** | Clinton Hill | `None` | `None` | None |
| **Pratt Institute - Memorial Hall Auditorium** | Clinton Hill | `None` | `None` | None |
| **Pupin Hall - Pupin Laboratories** | Morningside Heights | `None` | `None` | [Events Link](https://physics.columbia.edu/events.html) |
| **Recital Hall** | Willowbrook | `None` | `None` | None |
| **Recital Hall, College of Staten Island** | Willowbrook | `None` | `None` | None |
| **School of Visual Arts - SVA Theatre** | Chelsea | `None` | `None` | [Events Link](https://svatheatre.com/events/bfa-animation-faculty-show-tell-between-teaching-making/) |
| **Science & Engineering Lecture Hall** | Throgs Neck | `None` | `None` | None |
| **Science Building Lecture Hall** | Bayside | `None` | `None` | None |
| **Spiro Hall Lecture Halls, Wagner College** | Grymes Hill | `None` | `None` | None |
| **Springer Concert Hall** | Willowbrook | `None` | `None` | None |
| **Springer Concert Hall, College of Staten Island** | Willowbrook | `None` | `None` | None |
| **St. John's University - D'Angelo Center, Room 416** | Jamaica | `None` | `None` | None |
| **St. John's University - Little Theatre** | Jamaica | `None` | `None` | None |
| **St. John's University - Marillac Hall, Marillac Terrace** | Jamaica | `None` | `None` | None |
| **Terrace Room, The Regina S. Peruggi Academic Center, Kingsborough Community College** | Manhattan Beach | `None` | `None` | None |
| **The Auditorium, Alvin Johnson/J.M. Kaplan Hall, The New School** | Greenwich Village | `None` | `None` | None |
| **The Lovinger Theatre** | Bedford Park | `None` | `None` | None |
| **The New School - Amphitheater A404** | Greenwich Village | `None` | `None` | [Events Link](https://www.newschool.edu/campus-community/public-programs/) |
| **The New School - Starr Foundation Hall** | Greenwich Village | `None` | `None` | [Events Link](https://www.newschool.edu/campus-community/public-programs/) |
| **The New School - Theresa Lang Community and Student Center** | Greenwich Village | `None` | `None` | None |
| **The New School - Wollman Hall** | Greenwich Village | `None` | `None` | None |
| **Topfer Recital Hall, Brooklyn College** | Midwood | `None` | `None` | [Events Link](https://brooklyn.edu/tow/venues/topfer-recital-hall/events.html) |
| **Williamson Theatre** | Willowbrook | `None` | `None` | None |
| **Williamson Theatre, College of Staten Island** | Willowbrook | `None` | `None` | None |
| **Wollman Hall, Eugene Lang Building, The New School** | Greenwich Village | `None` | `None` | [Events Link](https://www.newschool.edu/campus-community/public-programs/) |
| **Yeshiva University - Belfer Hall** | Washington Heights | `None` | `None` | [Events Link](https://www.yu.edu/events) |
| **York College Little Theatre** | Jamaica | `None` | `None` | None |
| **York College Main Stage Theatre** | Jamaica | `None` | `None` | None |
| **York College Performing Arts Center** | Jamaica | `None` | `None` | None |

### concert halls (103 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Aaron Davis Hall** | Harlem | `KovZpZAdEJdA` | `None` | [Events Link](https://citycollegecenterforthearts.org/) |
| **Alewife Brewing** | Sunnyside | `not_found` | `None` | None |
| **American Legion Post 398** | Harlem | `not_found` | `None` | None |
| **Arlene Grocery** | Lower East Side | `KovZpZAFIv7A` | `api` | [Website](https://www.arlenesgrocerynyc.com/upcoming-events) |
| **Art Lab** | Livingston | `not_found` | `None` | None |
| **ArtSpace @ Staten Island Arts** | St. George | `not_found` | `None` | [Events Link](https://statenislandarts.org/events/) |
| **Arthouse Hotel New York City** | Upper West Side | `not_found` | `None` | [Website](https://www.arthousehotelnyc.com/) |
| **Astoria Performing Arts Center** | Astoria | `not_found` | `None` | None |
| **Bar 9** | Hell’s Kitchen | `not_found` | `None` | [Events Link](https://bar9ny.com/calendar-of-events/events) |
| **Bar Thalia** | Upper West Side | `not_found` | `None` | None |
| **Bay Street Tavern** | Stapleton | `not_found` | `None` | None |
| **Bierhaus** | Midtown East | `not_found` | `None` | [Events Link](https://www.bierhausnyc.com/events) |
| **Big Nose Kate's** | Rossville | `not_found` | `None` | None |
| **Bossa Nova Civic Club** | Bushwick | `not_found` | `None` | None |
| **Bronx House** | Pelham Parkway | `not_found` | `None` | [Events Link](https://www.bronxhouse.org/) |
| **Bronxlandia** | Hunts Point | `not_found` | `None` | [Website](https://bronxlandia.com/) |
| **Cafe Wha?** | Greenwich Village | `KovZpZAFteEA` | `None` | [Website](https://www.cafewha.com/tickets-reservations) |
| **Cantina Cumbancha** | Williamsburg | `not_found` | `None` | None |
| **Cape House** | Bushwick | `KovZpZAklnFA` | `None` | None |
| **Carnegie Hall** | Midtown Manhattan | `KovZpZA1dnJA` | `None` | [Website](https://www.carnegiehall.org/) |
| **Carroll Place** | Greenwich Village | `KovZpa2uGe` | `None` | [Website](https://carrollplacenyc.com/live-music/) |
| **Center Bar** | Upper West Side | `not_found` | `None` | None |
| **Center for the Arts at the College of Staten Island** | Willowbrook | `not_found` | `None` | None |
| **Centro Español de Queens** | Astoria | `not_found` | `None` | None |
| **Charlies Bar & Kitchen** | Mott Haven | `not_found` | `None` | None |
| **Chickering Hall** | NYC | `not_found` | `None` | None |
| **Club Groove** | Greenwich Village | `not_found` | `None` | [Events Link](https://www.clubgroovenyc.com/music-calendar) |
| **Cobra Club** | Bushwick | `not_found` | `None` | [Events Link](http://cobraclubbk.com/shows-and-events) |
| **College of Staten Island - Center for the Arts** | Willowbrook | `not_found` | `None` | None |
| **Cremorne Gardens** | NYC | `not_found` | `None` | None |
| **Deportee Studio** | West Bronx | `not_found` | `None` | [Events Link](https://www.deporteestudio.com/) |
| **Doc Hennigans** | West Brighton | `not_found` | `None` | None |
| **Dominie's** | Astoria | `not_found` | `None` | None |
| **East Williamsburg Music & Arts** | Williamsburg | `not_found` | `None` | None |
| **Feinstein's/54 Below** | Midtown | `not_found` | `None` | [Events Link](https://54below.org/events/vanessa-williams/) |
| **FirstLive Bushwick** | Bushwick | `not_found` | `None` | [Website](https://firstlive.us/) |
| **Fraunces Tavern** | Financial District | `rZ7HnEZ17AAuf` | `None` | [Website](https://www.frauncestavern.com/) |
| **Gold Sounds** | Bushwick | `not_found` | `None` | [Events Link](https://www.goldsounds.bar/events) |
| **Hamilton Park House Concerts** | New Brighton | `not_found` | `None` | None |
| **Happyfun Hideaway** | Bushwick | `not_found` | `None` | None |
| **Harlem Safe House Jazz Parlor** | Harlem | `not_found` | `None` | None |
| **Haugen Hall Theatre at Staten Island Academy** | Todt Hill | `not_found` | `None` | [Events Link](https://www.statenislandacademy.org/about/event-calendar) |
| **Heaven Can Wait** | East Village | `Z7r9jZa7Pn` | `None` | [Website](https://heavencanwaitnyc.com/calendar/) |
| **Hill Country Barbecue** | Chelsea | `not_found` | `None` | [Events Link](https://www.hillcountry.com/hill-country-live-ny/) |
| **Hub 17** | Stapleton | `not_found` | `None` | None |
| **Isola** | Bushwick | `not_found` | `None` | None |
| **Isola Brooklyn** | Williamsburg | `not_found` | `None` | None |
| **Jack Jones** | Astoria | `not_found` | `None` | None |
| **Joe's Garage Bar** | Astoria | `not_found` | `None` | None |
| **Jupiter Disco** | Bushwick | `not_found` | `None` | [Events Link](https://www.jupiterdisco.com/) |
| **Kettle Black** | West Brighton | `not_found` | `None` | [Events Link](https://www.kettleblackbar.com/events) |
| **La Conga** | NYC | `not_found` | `None` | None |
| **Littlefield** | Gowanus | `KovZpZAFEFJA` | `None` | [Website](https://littlefieldnyc.com/calendar) |
| **Londel's Supper Club** | Harlem | `not_found` | `None` | None |
| **Lot 45** | Bushwick | `not_found` | `None` | None |
| **Lucky Dog** | Williamsburg | `not_found` | `None` | None |
| **Mad Tropical** | Bushwick | `not_found` | `None` | None |
| **Manderlay Bar** | Chelsea | `not_found` | `None` | [Events Link](https://mckittrickhotel.com/events/) |
| **Manhattan School of Music - Miller Recital Hall** | Morningside Heights | `not_found` | `None` | None |
| **Marjorie Eliot's Parlor Jazz** | Harlem | `not_found` | `None` | None |
| **Metropolitan Opera House** | Upper West Side | `not_found` | `None` | [Events Link](https://www.metopera.org/season/events/) |
| **Morgan Library & Museum - Gilder Lehrman Hall** | Murray Hill | `not_found` | `None` | [Events Link](https://www.themorgan.org/drawing-institute/events) |
| **Northwell Health Jones Beach Theater** | Wantagh | `not_found` | `api` | None |
| **Otto’s Shrunken Head** | East Village | `not_found` | `None` | [Events Link](https://ottosshrunkenhead.com/pages/events.php) |
| **Paddy Reilly’s Music Bar** | Murray Hill / Kips Bay | `not_found` | `None` | None |
| **Paragon** | Williamsburg | `not_found` | `None` | [Website](https://paragon.nyc/) |
| **Parlor Entertainment** | Harlem | `not_found` | `None` | None |
| **Public Arts** | Lower East Side | `KovZpZA6eItA` | `None` | None |
| **Rambling House** | Woodlawn | `not_found` | `None` | [Website](https://www.ramblinghousenyc.com/) |
| **Rockwood Music Hall** | Lower East Side | `KovZ917A9F0` | `None` | [Website](https://rockwoodmusichall.com/) |
| **Rose Theater** | Upper West Side | `KovZpZA11JJA` | `None` | [Events Link](https://jazz.org/watch-listen-discover/program-archive/) |
| **Secret Pour** | Bushwick | `not_found` | `None` | [Events Link](https://www.secretpour.com/events) |
| **Sidewalk Cafe** | East Village | `not_found` | `None` | None |
| **Singlecut Beersmiths** | Astoria | `not_found` | `None` | [Website](https://singlecut.com/) |
| **Sleepwalk** | Bushwick | `not_found` | `None` | [Website](https://sleepwalk.nyc/) |
| **Spotlight Studios** | West Brighton | `not_found` | `None` | None |
| **Stathakion Cultural Center** | Astoria | `not_found` | `None` | None |
| **Sunnyvale** | East Williamsburg | `KovZpZAktJJA` | `None` | None |
| **Superior Ingredients** | Williamsburg | `KovZ917AYxu` | `None` | [Website](https://www.superioringredients.com/) |
| **TBA Brooklyn** | Williamsburg | `Z7r9jZaAWc` | `None` | [Website](https://tbanyc.com/) |
| **Tavern Concerts** | Richmondtown | `Z7r9jZaeRT` | `None` | None |
| **The Appel Room** | Upper West Side | `KovZ917A2q7` | `None` | [Events Link](https://jazz.org/watch-listen-discover/program-archive/) |
| **The Astor Room** | Astoria | `not_found` | `None` | None |
| **The Atrium Gallery at Staten Island Academy** | Todt Hill | `not_found` | `None` | [Events Link](https://www.statenislandacademy.org/about/event-calendar) |
| **The Bitter End** | Greenwich Village | `ZFr9jZaA6a` | `None` | [Website](https://bitterend.com/#/events) |
| **The Bronx Beer Hall** | Belmont | `not_found` | `None` | [Events Link](https://thebronxbeerhall.com/events) |
| **The Bronx Brewery** | Port Morris | `not_found` | `None` | [Website](https://thebronxbrewery.com/) |
| **The Curly Wolf** | Great Kills | `not_found` | `None` | None |
| **The Dog and Duck** | Sunnyside | `not_found` | `None` | None |
| **The Hop Shoppe** | Stapleton | `not_found` | `None` | [Website](https://thehopshoppe.com/) |
| **The Juilliard School - Morse Hall** | Upper West Side | `not_found` | `None` | [Events Link](https://www.juilliard.edu/events.html) |
| **The Juilliard School - Paul Recital Hall** | Upper West Side | `not_found` | `None` | [Events Link](https://www.juilliard.edu/campus-life/performance-facilities/paul-recital-hall/events.html) |
| **The Juilliard School - Peter Jay Sharp Theater** | Upper West Side | `not_found` | `None` | [Events Link](https://www.juilliard.edu/campus-life/performance-venues/peter-jay-sharp-theater/events.html) |
| **The Keep** | Bushwick | `not_found` | `None` | None |
| **The Letlove Inn** | Astoria | `not_found` | `None` | None |
| **The Red Lion** | Greenwich Village | `not_found` | `None` | [Website](https://www.redlionnyc.com/) |
| **The Rock House** | Prince's Bay | `not_found` | `None` | None |
| **The Strand Smokehouse** | Astoria | `not_found` | `None` | None |
| **The Vino Theater** | East Williamsburg | `not_found` | `None` | None |
| **The Wallace Lounge** | Upper West Side | `not_found` | `None` | None |
| **The Williamsburg Opera House** | Williamsburg | `not_found` | `None` | None |
| **Ulysses’** | Financial District | `not_found` | `None` | [Events Link](https://www.ulyssesnyc.com/events/) |
| **Westerleigh Park** | Westerleigh | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |

### rock music venues (85 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Alan Block's Sandal Shop** | Greenwich Village | `not_found` | `None` | None |
| **American Bandstand Theater** | Midtown | `not_found` | `None` | None |
| **Area** | Tribeca | `KovZpZAktAFA` | `None` | None |
| **Arlene's Grocery** | Lower East Side | `KovZpZAFIv7A` | `None` | [Website](http://arlenesgrocery.net/) |
| **B.B. King Blues Club & Grill** | Times Square | `KovZpZAEAE1A` | `None` | None |
| **Bond's International Casino** | Times Square | `not_found` | `None` | None |
| **Bop City** | Midtown | `not_found` | `None` | None |
| **Brill Building** | Midtown | `not_found` | `None` | [Events Link](https://brillbuilding.com/) |
| **Brownies** | East Village | `not_found` | `None` | None |
| **CBGB and OMFUG (CBGB's)** | East Village | `not_found` | `None` | None |
| **Cafe Bizarre** | Greenwich Village | `not_found` | `None` | None |
| **Cafe Feenjon** | Greenwich Village | `not_found` | `None` | None |
| **Cafe Figaro** | Greenwich Village | `not_found` | `None` | [Events Link](https://www.figaronyc.com/) |
| **Cafe Society** | West Village | `not_found` | `None` | None |
| **Cafe au Go Go** | Greenwich Village | `not_found` | `None` | None |
| **Cheetah** | Midtown | `not_found` | `None` | None |
| **Chelsea Hotel** | Chelsea | `not_found` | `None` | [Events Link](https://www.hotelchelsea.com/events) |
| **Coney Island High** | East Village | `Z6r9jZ7F1e` | `None` | None |
| **Copacapana** | Midtown East | `not_found` | `None` | [Website](https://copacabanany.com/) |
| **Danceteria** | Chelsea | `not_found` | `None` | None |
| **El Morocco** | Midtown East | `not_found` | `None` | None |
| **Electric Lady Studio** | Greenwich Village | `not_found` | `None` | [Website](https://electricladystudios.com/) |
| **Fez** | NoHo | `ZFr9jZ7k7k` | `None` | None |
| **Fillmore East** | East Village | `not_found` | `None` | None |
| **Fox Theater** | Brooklyn | `not_found` | `None` | None |
| **Gerde's Folk City** | Greenwich Village | `not_found` | `None` | None |
| **Great Gildersleeves** | East Village | `not_found` | `None` | None |
| **Highline Ballroom** | Chelsea | `not_found` | `None` | None |
| **Hurrah** | Upper West Side | `not_found` | `None` | None |
| **Hurricane** | Midtown | `Z7r9jZadyx` | `None` | None |
| **Izzy Young's Folklore Center** | Greenwich Village | `not_found` | `None` | None |
| **Kenny's Castaways** | Greenwich Village | `not_found` | `None` | None |
| **King Tut's Wah Wah Hut** | East Village | `not_found` | `None` | None |
| **Latin Quarter** | Midtown | `not_found` | `None` | None |
| **Lismar Lounge** | East Village | `not_found` | `None` | None |
| **Lone Star Cafe** | Greenwich Village | `not_found` | `None` | None |
| **Lone Star Roadhouse** | Midtown | `not_found` | `None` | None |
| **Luna Lounge** | Lower East Side | `not_found` | `None` | None |
| **Max's Kansas City** | Gramercy | `not_found` | `None` | None |
| **Maxwell's** | Hoboken | `KovZpZAav7JA` | `None` | None |
| **Mills Tavern** | Greenwich Village | `not_found` | `None` | None |
| **Ondine** | Midtown East | `not_found` | `None` | None |
| **Paradise Cabaret** | Midtown | `not_found` | `None` | None |
| **Paradise Garage** | Hudson Square | `not_found` | `None` | None |
| **Playstation Theater** | Times Square | `not_found` | `None` | None |
| **Roseland Ballroom** | Midtown | `KovZpZAF6AJA` | `None` | None |
| **Save the Robots** | East Village | `not_found` | `None` | None |
| **Shea Stadium** | Queens | `KovZpZA1dktA` | `None` | None |
| **Steve Paul's The Scene** | Midtown | `not_found` | `None` | None |
| **The Academy** | Midtown | `KovZpZAJF1aA` | `None` | None |
| **The Bottom Line** | Greenwich Village | `not_found` | `None` | None |
| **The Cat Club** | Greenwich Village | `not_found` | `None` | None |
| **The Cock 'n Bull** | Greenwich Village | `not_found` | `None` | None |
| **The Cotton Club** | Harlem / Midtown | `not_found` | `None` | [Website](http://www.cottonclub-newyork.com/) |
| **The Dom** | East Village | `not_found` | `None` | None |
| **The Electric Circus** | East Village | `not_found` | `None` | None |
| **The Garrick Theater** | Greenwich Village | `not_found` | `None` | None |
| **The Gaslight Cafe** | Greenwich Village | `not_found` | `None` | None |
| **The Hit Factory** | Midtown / Hell's Kitchen | `not_found` | `None` | None |
| **The Kettle of Fish** | Greenwich Village | `not_found` | `None` | [Events Link](http://kettleoffishnyc.com/) |
| **The Limelight (Chelsea)** | Chelsea | `not_found` | `None` | None |
| **The Limelight (Village)** | Greenwich Village | `not_found` | `None` | None |
| **The Mercer Arts Center** | Greenwich Village | `not_found` | `None` | None |
| **The Mudd Club** | Tribeca | `not_found` | `None` | None |
| **The Night Owl Cafe** | Greenwich Village | `not_found` | `None` | None |
| **The Other End** | Greenwich Village | `not_found` | `None` | None |
| **The Peppermint Lounge** | Midtown | `not_found` | `None` | None |
| **The Power Station (Avatar)** | Hell's Kitchen | `not_found` | `None` | None |
| **The Pyramid Club** | East Village | `not_found` | `None` | None |
| **The Rainbow Room** | Midtown | `KovZpZA1JelA` | `None` | [Website](https://www.rainbowroom.com/) |
| **The Record Plant** | Midtown | `not_found` | `None` | None |
| **The Saint** | East Village | `Z7r9jZaeJV` | `None` | None |
| **The Stork Club** | Midtown | `not_found` | `None` | None |
| **The Tunnel** | Chelsea | `rZ7HnEZ17q__V` | `None` | None |
| **The Village Gate** | Greenwich Village | `not_found` | `None` | None |
| **The White Horse Tavern** | West Village | `not_found` | `None` | [Events Link](https://www.whitehorsetavern1880.com/events) |
| **The World** | East Village | `KovZpab_we` | `None` | None |
| **Tin Angel** | Greenwich Village | `not_found` | `None` | None |
| **Tramp's** | Gramercy / Chelsea | `ZFr9jZeF77` | `None` | None |
| **Trax** | Upper West Side | `not_found` | `None` | None |
| **Trude Heller's** | Greenwich Village | `not_found` | `None` | None |
| **Union Hall** | Park Slope | `rZ7HnEZa7Gl` | `None` | [Website](https://www.unionhallny.com/) |
| **Wetlands Preserve** | Tribeca | `not_found` | `None` | None |
| **Xenon** | Midtown | `not_found` | `None` | None |
| **Zanzibar** | Midtown | `not_found` | `None` | None |

### live music venues (83 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **00:00** | Long Island City | `not_found` | `None` | None |
| **7B Horseshoe Bar (Vazacs)** | East Village | `not_found` | `None` | None |
| **Astoria Bier & Cheese** | Astoria | `not_found` | `None` | [Website](https://www.astoriabierandcheese.com/) |
| **Astoria World Manor** | Astoria | `KovZpZA1d1IA` | `None` | None |
| **Attaboy** | Lower East Side | `not_found` | `None` | None |
| **Azara Kitchen** | Harlem | `not_found` | `None` | None |
| **Barramundi** | Lower East Side | `not_found` | `None` | None |
| **Bel Aire Diner** | Astoria | `not_found` | `None` | [Events Link](https://www.belairediner.nyc/events) |
| **Billie's Black** | Harlem | `not_found` | `None` | None |
| **Blue & Gold Tavern** | East Village | `not_found` | `None` | None |
| **Boro Hotel** | Long Island City | `not_found` | `None` | None |
| **Brewster LIC** | Long Island City | `not_found` | `None` | [Website](https://brewsterlic.com/) |
| **Cherry Tavern** | East Village | `not_found` | `None` | None |
| **Clandestino** | Lower East Side | `not_found` | `None` | None |
| **Continental Lounge** | East Village | `not_found` | `None` | None |
| **Court Square Theater** | Long Island City | `not_found` | `None` | None |
| **Cypress Cafe** | Ridgewood | `not_found` | `None` | None |
| **Diamond Dogs** | Astoria | `not_found` | `None` | None |
| **Dutch Kills** | Long Island City | `not_found` | `None` | [Events Link](https://www.dutchkillsbar.com) |
| **Eastwood** | Lower East Side | `not_found` | `None` | None |
| **Evil Twin Brewing NYC** | Ridgewood | `not_found` | `None` | [Events Link](https://eviltwin.nyc/events) |
| **Farafina Cafe & Lounge Harlem** | Harlem | `not_found` | `None` | None |
| **Fig 19** | Lower East Side | `not_found` | `None` | None |
| **Gottscheer Hall** | Ridgewood | `not_found` | `None` | [Events Link](http://gottscheerhall.com/events) |
| **Hellenic Cultural Center** | Astoria | `not_found` | `None` | [Events Link](https://www.hellenicculturalcenter.org/) |
| **Holiday Cocktail Lounge** | East Village | `not_found` | `None` | None |
| **International Bar** | East Village | `not_found` | `None` | None |
| **Irish Rover** | Astoria | `not_found` | `None` | None |
| **Jack Jones Gastropub** | Astoria | `not_found` | `None` | None |
| **Josie's Bar** | East Village | `not_found` | `None` | [Events Link](https://www.josiesnyc.com/events.html) |
| **Kelly's Sports Bar** | East Village | `not_found` | `None` | [Events Link](https://www.kellysnyc.com/calendar) |
| **Kind Regards** | Lower East Side | `not_found` | `None` | [Website](https://www.kindregardsnyc.com/) |
| **Little Branch** | Greenwich Village | `not_found` | `None` | None |
| **Lucky** | East Village | `Z7r9jZaAqr` | `None` | None |
| **Madam Marie's** | Astoria | `not_found` | `None` | [Website](https://www.madammaries.com/) |
| **Make Believe** | Lower East Side | `rZ7HnEZ17fOZA` | `None` | [Website](https://www.sixtyhotels.com/sixty-les/eat-drink/make-believe/) |
| **Marshall Stack** | Lower East Side | `not_found` | `None` | None |
| **Max Fish** | Lower East Side | `not_found` | `None` | None |
| **McSorley's Old Ale House** | East Village | `not_found` | `None` | [Events Link](https://mcsorleysoldalehouse.nyc/events.html) |
| **Mona's** | East Village | `not_found` | `None` | None |
| **Niagara** | East Village | `KovZpZA7AvlA` | `None` | [Events Link](http://www.niagaranyc.com/) |
| **No Fun** | Lower East Side | `Z7r9jZaAdP` | `None` | [Events Link](http://nofun-nyc.com) |
| **Parlor Jazz at Marjorie Eliot's** | Washington Heights | `not_found` | `None` | None |
| **Plowshares** | Morningside Heights | `not_found` | `None` | [Events Link](https://plowsharescoffee.com/) |
| **Queens Brewery** | Ridgewood | `not_found` | `None` | None |
| **R-Bar** | Lower East Side | `not_found` | `None` | None |
| **Renaissance Ballroom & Casino** | Harlem | `not_found` | `None` | None |
| **Ridgewood Presbyterian Church** | Ridgewood | `not_found` | `None` | None |
| **Rivercrest** | Astoria | `not_found` | `None` | None |
| **Rue-B** | East Village | `not_found` | `None` | None |
| **Sanfords Restaurant** | Astoria | `not_found` | `None` | None |
| **Savoy Ballroom** | Harlem | `not_found` | `None` | None |
| **Secret Loft** | Bushwick | `not_found` | `None` | None |
| **Showman's Cafe** | Harlem | `not_found` | `None` | None |
| **Smalls Paradise** | Harlem | `not_found` | `None` | None |
| **Smoke Jazz Club** | Morningside Heights | `not_found` | `None` | [Events Link](https://smokejazz.com/) |
| **Sophie's** | East Village | `not_found` | `None` | None |
| **Sound River Studios** | Long Island City | `not_found` | `None` | [Website](https://www.soundriverstudios.com/) |
| **St. Jerome's** | Lower East Side | `not_found` | `None` | None |
| **St. Nick's Pub** | Sugar Hill | `not_found` | `None` | None |
| **Stone Circle Theatre** | Ridgewood | `Z7r9jZaA-F` | `None` | [Website](https://www.stonecircletheatre.org/) |
| **Sugar Hill Supper Club** | Harlem | `not_found` | `None` | None |
| **Sweet Afton** | Astoria | `not_found` | `None` | [Events Link](https://www.sweetaftonbar.com/events/) |
| **The Amphitheater at Coney Island Boardwalk** | Coney Island | `not_found` | `None` | [Website](https://www.coneyislandlive.com/) |
| **The Cast** | East Village | `not_found` | `None` | None |
| **The Corner Lounge Bistro** | Harlem | `not_found` | `None` | None |
| **The Ditty** | Astoria | `not_found` | `None` | [Events Link](https://www.thedittybar.com/) |
| **The Edge LIC** | Long Island City | `not_found` | `None` | None |
| **The Flower Shop** | Lower East Side | `not_found` | `None` | [Events Link](https://theflowershopnyc.com/events) |
| **The Footlight** | Ridgewood | `not_found` | `None` | None |
| **The Harlem Alhambra** | Harlem | `not_found` | `None` | None |
| **The Last Word** | Astoria | `not_found` | `None` | None |
| **The Lenox Lounge** | Harlem | `not_found` | `None` | None |
| **The Library** | East Village | `ZFr9jZFAkF` | `None` | None |
| **The Local Bar** | Astoria | `not_found` | `None` | None |
| **The Marquee at Astoria** | Astoria | `not_found` | `None` | None |
| **The Meadows** | Richmond Hill | `rZ7HnEZ17faug` | `None` | None |
| **The Renaissance** | Harlem | `Z7r9jZaAfM` | `None` | None |
| **The Varnish** | Lower East Side | `not_found` | `None` | None |
| **Trinity Reformed Church** | Ridgewood | `not_found` | `None` | None |
| **Unruly Collective** | Bushwick | `not_found` | `None` | None |
| **Waikiki Wally's** | East Village | `not_found` | `None` | None |
| **Welcome to the Johnson's** | Lower East Side | `not_found` | `None` | None |

### sports arenas stadiums (77 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **APEX Arena** | Bedford Park | `not_found` | `None` | [Events Link](https://lehmanathletics.com/calendar.aspx) |
| **ARC Arena** | Flatiron District | `not_found` | `None` | [Events Link](https://athletics.baruch.cuny.edu/calendar) |
| **Abe Stark Sports Center** | Coney Island | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Alley Pond Tennis Center** | Alley Pond Park | `not_found` | `None` | [Events Link](https://www.alleypondtenniscenter.com/) |
| **Aviator Sports and Events Center** | Marine Park | `KovZpZAdt7aA` | `None` | [Events Link](https://www.aviatorsports.com/sports/program-finder/) |
| **BMCC Gymnasium** | Tribeca | `not_found` | `None` | [Events Link](https://bmccathletics.com/) |
| **Bronx Community College Alumni Gymnasium** | University Heights | `not_found` | `None` | None |
| **Bushwick Inlet Park** | Williamsburg | `not_found` | `None` | None |
| **Canarsie Park** | Canarsie | `not_found` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **City Ice Pavilion** | Long Island City | `not_found` | `None` | None |
| **Coffey Field** | Fordham | `KovZpZAtvE7A` | `None` | [Events Link](https://fordhamsports.com/facilities/jack-coffey-field/1) |
| **Concourse Plaza Multiplex Cinemas** | Concourse | `KovZpZAaavEA` | `None` | None |
| **Cunningham Tennis Center** | Cunningham Park | `not_found` | `None` | [Events Link](https://cunninghamtennis.com/events/) |
| **Downtown Tennis Club** | Bay Ridge | `not_found` | `None` | None |
| **East River Park Track** | Lower East Side | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Ebbets Field (historical)** | Crown Heights, Brooklyn | `not_found` | `None` | None |
| **Fort Greene Park Tennis Courts** | Fort Greene | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Fort Totten Park** | Bayside | `not_found` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Generoso Pope Athletic Complex** | Brooklyn Heights | `not_found` | `None` | None |
| **Giants Stadium** | East Rutherford | `not_found` | `None` | None |
| **Health and Physical Education Complex** | Jamaica | `not_found` | `None` | [Events Link](https://yorkathletics.com/calendar.aspx) |
| **Hostos Community College Gymnasium** | Mott Haven | `not_found` | `None` | None |
| **Hunter College Sportsplex** | Upper East Side | `not_found` | `None` | None |
| **John A. Paulson Center** | Greenwich Village | `not_found` | `None` | [Website](https://www.nyu.edu/about/visitor-information/john-a-paulson-center.html) |
| **Joseph Yancey Track and Field** | Concourse | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Kaiser Park** | Coney Island | `not_found` | `None` | None |
| **Kips Bay Boys & Girls Club Ice Rink** | Castle Hill | `not_found` | `None` | None |
| **Kissena Park Tennis Courts** | Kissena Park | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **LaGuardia Community College Gymnasium** | Long Island City | `not_found` | `None` | [Events Link](https://www.laguardia.edu/events/?ical=1) |
| **LeFrak Center at Lakeside** | Prospect Park | `not_found` | `None` | [Events Link](https://www.prospectpark.org/visit/places-to-go/lefrak-center-lakeside/events.html) |
| **LeFrak Center at Lakeside Prospect Park** | Prospect Lefferts Gardens | `not_found` | `None` | [Events Link](https://www.prospectpark.org/visit/places-to-go/lefrak-center-lakeside/events.html) |
| **Lincoln Terrace Park Tennis Courts** | Crown Heights | `not_found` | `None` | None |
| **MCU Park (Maimonides Park)** | Coney Island, Brooklyn | `not_found` | `None` | None |
| **Macombs Dam Park** | Concourse | `KovZpZA1klkA` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Manhattan Plaza Racquet Club** | Hell's Kitchen | `not_found` | `None` | [Events Link](https://www.mprctennis.com/) |
| **Max Stern Athletic Center** | Washington Heights | `not_found` | `None` | [Events Link](https://yu.edu/events) |
| **McCarren Park Tennis Courts** | Williamsburg | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **MetLife Stadium** | East Rutherford | `not_found` | `None` | [Events Link](https://www.metlifestadium.com/events/fifa-world-cup-2026) |
| **Metropolitan Oval** | Maspeth | `not_found` | `None` | None |
| **Nat Holman Gymnasium** | Hamilton Heights | `not_found` | `None` | None |
| **New York Tennis Club** | Throgs Neck | `not_found` | `None` | [Events Link](https://www.newyorktennisclub.com/) |
| **Ohio Field** | University Heights | `not_found` | `None` | None |
| **Paerdegat Basin Park** | Canarsie | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Pelham Bay Park** | Pelham Bay | `Z7r9jZaeNA` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Polo Grounds (historical)** | Washington Heights, Manhattan | `not_found` | `None` | None |
| **Pratt ARC Building** | Clinton Hill | `not_found` | `None` | [Events Link](https://goprattgo.com/calendar) |
| **Pratt Institute Activities Resource Center** | Clinton Hill | `not_found` | `None` | None |
| **Prospect Park Tennis Center** | Prospect Park | `not_found` | `None` | [Events Link](https://www.prospectpark.org/visit/places-to-go/tennis-center/events.html) |
| **Queensborough Community College Gymnasium** | Bayside | `not_found` | `None` | [Events Link](https://www.qcc.cuny.edu/athletics/rfk-gymnasium.html/events) |
| **Recreation and Wellness Center** | Brownsville | `not_found` | `None` | None |
| **Red Bull Arena** | Harrison, New Jersey | `not_found` | `None` | None |
| **Red Hook Park** | Red Hook | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Red Hook Recreation Area** | Red Hook | `not_found` | `None` | None |
| **Reinhart Field** | Throggs Neck | `not_found` | `None` | None |
| **Richmond County Bank Ballpark at St. George** | St. George, Staten Island | `KovZpapJHe` | `None` | [Events Link](https://ferryhawks.com/sports/baseball/schedule) |
| **Riesenberg Hall Gymnasium** | Throggs Neck | `not_found` | `None` | None |
| **Riverdale Tennis Center** | Riverdale | `not_found` | `None` | None |
| **Rose Hill Gymnasium** | Rose Hill | `KovZpaFNme` | `None` | [Events Link](https://fordhamsports.com/facilities/rose-hill-gymnasium/1) |
| **SIUH Community Park** | St. George | `not_found` | `None` | [Events Link](https://www.ferryhawks.com/sports/baseball/schedule) |
| **Silver Lake Tennis Program** | Silver Lake | `not_found` | `None` | None |
| **Sky Rink at Chelsea Piers** | Chelsea | `not_found` | `None` | [Website](https://thechelseapiers.com/sky-rink/) |
| **South Brother Island** | Bronx | `not_found` | `None` | None |
| **Sportime at Randall's Island** | Randall's Island | `not_found` | `None` | None |
| **Sports and Recreation Center** | Willowbrook | `not_found` | `None` | None |
| **Sunnyside Garden Arena (historical)** | Sunnyside, Queens | `not_found` | `None` | None |
| **Tenniscape** | Bay Ridge | `not_found` | `None` | None |
| **The Doghouse** | Hell's Kitchen | `not_found` | `None` | None |
| **The Graveyard** | Long Island City | `not_found` | `None` | [Events Link](https://www.thegraveyardnyc.com/) |
| **The Hill Center** | Clinton Hill | `KovZpZA167IA` | `None` | None |
| **Thomas Jefferson Park** | East Harlem | `not_found` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Tottenville Racquet Club** | Tottenville | `not_found` | `None` | None |
| **USTA Billie Jean King National Tennis Center** | Flushing, Queens | `not_found` | `None` | [Events Link](https://www.ntc.usta.com/tennis-programs-and-camps/adult-programs) |
| **Van Cortlandt Park Tennis Courts** | Van Cortlandt Park | `not_found` | `None` | None |
| **Vanderbilt Tennis Club** | Midtown | `not_found` | `None` | [Website](https://www.vanderbilttennisclub.com/) |
| **Westerleigh Tennis Club** | Westerleigh | `not_found` | `None` | None |
| **York College Health & Physical Education Complex** | Jamaica | `not_found` | `None` | None |
| **Yorkville Tennis Club** | Yorkville | `not_found` | `None` | [Events Link](https://yorkvilletennisclub.com/juniorprograms/) |

### classical music venues (55 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Alice Tully Hall** | Upper West Side | `KovZpZAkJFkA` | `None` | [Website](http://www.lincolncenter.org/) |
| **Arete Venue and Gallery** | Greenpoint | `not_found` | `None` | None |
| **Baisley Powell Elebash Recital Hall** | Greenwich Village | `not_found` | `None` | [Events Link](https://www.gc.cuny.edu/about-graduate-center/facilities-and-services/room-reservations/elebash-recital-hall/events.html) |
| **Baryshnikov Arts Center - Howard Gilman Performance Space** | Hell's Kitchen | `not_found` | `None` | [Events Link](https://bacnyc.org/) |
| **Brooklyn Public Library - Central Library** | Prospect Heights | `not_found` | `None` | [Events Link](https://www.bklynlibrary.org/calendar/citizenship-class-central-library-business-20260217-0930am) |
| **Brooklyn Public Library, Dweck Center** | Prospect Heights | `not_found` | `None` | None |
| **Carla Bossi-Comelli Studio** | Morningside Heights | `not_found` | `None` | None |
| **Cathedral of St. John the Divine** | Morningside Heights | `KovZpZAat7aA` | `None` | [Events Link](https://www.stjohndivine.org/events.html) |
| **Charles Myers Recital Hall** | Morningside Heights | `not_found` | `None` | None |
| **Christ Church, Cobble Hill** | Cobble Hill | `KovZpapVUe` | `None` | None |
| **Church of St. Philip Neri** | Bedford Park | `not_found` | `None` | [Events Link](https://stphilipneribronx.org/events) |
| **Corpus Christi Church** | Morningside Heights | `not_found` | `None` | None |
| **David A. Rahm Hall** | Morningside Heights | `not_found` | `None` | None |
| **Elebash Recital Hall** | Midtown | `not_found` | `None` | [Events Link](https://www.gc.cuny.edu/about-graduate-center/facilities-and-services/room-reservations/elebash-recital-hall/events.html) |
| **Elebash Recital Hall - CUNY Graduate Center** | Midtown | `not_found` | `None` | [Events Link](https://www.gc.cuny.edu/about-graduate-center/facilities-and-services/room-reservations/elebash-recital-hall/events.html) |
| **Ernst C. Stiefel Hall** | Greenwich Village | `not_found` | `None` | None |
| **First Presbyterian Church of Forest Hills** | Forest Hills | `not_found` | `None` | None |
| **Florence Gould Hall at FIAF** | Midtown East | `not_found` | `None` | [Events Link](https://fiaf.org/venue/florence-gould-hall/events.html) |
| **Florence Gould Hall at French Institute Alliance Française (FIAF)** | Midtown East | `not_found` | `None` | [Events Link](https://fiaf.org/events.html) |
| **Glassbox Theater** | Greenwich Village | `not_found` | `None` | None |
| **Good Shepherd-Faith Presbyterian Church** | Lincoln Square | `not_found` | `None` | None |
| **Grace Church Brooklyn Heights** | Brooklyn Heights | `not_found` | `None` | [Events Link](https://gracebrooklyn.org/) |
| **Green-Wood Cemetery Catacombs (The Angel's Share)** | Greenwood | `not_found` | `None` | [Website](https://www.green-wood.com/) |
| **Groupmuse** | Various | `not_found` | `None` | [Events Link](https://www.groupmuse.com/events/16147-artek-presents-mozart-and-a-woman-of-genius) |
| **Hostos Center for the Arts & Culture** | Mott Haven | `not_found` | `None` | [Website](https://www.hostoscenter.org/) |
| **Jazz at Lincoln Center: Rose Theater** | Midtown | `not_found` | `None` | [Events Link](https://jazz.org/concerts-events/calendar/) |
| **Klavierhaus** | Hell's Kitchen | `not_found` | `None` | [Events Link](https://event.klavierhaus.com/k/calendar) |
| **Lowlands Bar** | Gowanus | `not_found` | `None` | None |
| **MOMA Summer Contemporary Classical Music Garden** | Midtown | `not_found` | `None` | [Events Link](http://www.moma.org/calendar/exhibitions/history) |
| **Mannes School of Music - Mannes Concert Hall** | Greenwich Village | `not_found` | `None` | [Events Link](https://www.newschool.edu/mannes/events/) |
| **NYU Steinhardt - Black Box Theatre** | Greenwich Village | `not_found` | `None` | None |
| **OPERA America's National Opera Center** | Chelsea | `not_found` | `None` | [Events Link](https://www.operaamerica.org/programs/events/opera-america-salutes/) |
| **Provincetown Playhouse** | Greenwich Village | `not_found` | `None` | None |
| **Queens College Concert Hall & Colden Auditorium** | Flushing | `not_found` | `None` | [Website](http://www.coldencenter.org/) |
| **Spectrum** | Brooklyn Navy Yard | `KovZpZA1JvFA` | `None` | [Events Link](http://spectrumnyc.com/events.html) |
| **St. Alban's Park** | St. Albans | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **St. Ann & the Holy Trinity Church** | Brooklyn Heights | `Z7r9jZaeBF` | `None` | [Events Link](https://www.stannholytrinity.org/eventsnew) |
| **St. John's University - Great Lawn** | Jamaica | `not_found` | `None` | [Events Link](https://www.stjohns.edu/academics/programs?level%5B146%5D=146) |
| **St. Malachy's Church** | Midtown | `not_found` | `None` | [Events Link](https://actorschapel.org/events) |
| **St. Paul's Chapel at Columbia University** | Morningside Heights | `not_found` | `None` | [Events Link](https://religiouslife.columbia.edu/st-pauls-chapel/events.html) |
| **St. Paul's Evangelical Lutheran Church** | Parkchester | `not_found` | `None` | [Website](https://stpaulswilliamsburg.com/) |
| **SubCulture** | NoHo | `Z7r9jZaerq` | `None` | None |
| **The Cell Theatre** | Chelsea | `not_found` | `None` | [Events Link](https://www.thecelltheatre.org/events) |
| **The Church-in-the-Gardens** | Forest Hills | `not_found` | `None` | [Website](https://churchinthegardens.org/) |
| **The Community Church of Douglaston** | Douglaston | `not_found` | `None` | None |
| **The Crypt under The Church of the Intercession (The Crypt Sessions)** | Harlem | `not_found` | `None` | None |
| **The Green-Wood Cemetery Catacombs** | Greenwood Heights | `not_found` | `None` | [Website](https://www.green-wood.com/) |
| **The Greene Space at WNYC** | Hudson Square | `not_found` | `None` | [Events Link](https://thegreenespace.org/events) |
| **The Greene Space at WNYC & WQXR** | Hudson Square | `not_found` | `None` | [Events Link](https://thegreenespace.org/events) |
| **The High School of Fashion Industries** | Chelsea | `not_found` | `None` | [Events Link](https://www.hsfi.nyc/events) |
| **The New School - Tishman Auditorium** | Greenwich Village | `ZFr9jZFAvk` | `None` | None |
| **The Solomon Gadles Mikowsky Recital Hall** | Morningside Heights | `not_found` | `None` | None |
| **The William R. and Irene D. Miller Recital Hall** | Morningside Heights | `not_found` | `None` | None |
| **Third Street Music School Settlement** | East Village | `not_found` | `None` | [Events Link](https://www.thirdstreet.nyc/events) |
| **WMP Concert Hall** | NoMad | `not_found` | `None` | None |

### jazz clubs (47 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **449 LA SCAT** | Harlem | `not_found` | `None` | None |
| **Antique Garage Restaurant** | SoHo | `not_found` | `None` | [Events Link](http://www.antiquegaragesoho.com/) |
| **Bar Next Door** | Greenwich Village | `not_found` | `None` | [Events Link](http://lalanternacaffe.com/barnextdoor.html) |
| **Barbès** | Park Slope | `not_found` | `None` | [Events Link](https://www.barbesbrooklyn.com/events) |
| **Cafe Bohemia** | West Village | `not_found` | `None` | [Events Link](https://www.cafebohemiany.com/events) |
| **Cellar Dog** | West Village | `not_found` | `None` | [Events Link](https://www.cellardog.net/events) |
| **Cleopatra’s Needle** | Upper West Side | `not_found` | `None` | [Events Link](http://www.cleopatrasneedleny.com/holidayevents.html) |
| **Close Up** | Lower East Side | `not_found` | `None` | [Events Link](https://www.closeupnyc.com/calendar) |
| **Club BonaFide** | Midtown Manhattan | `not_found` | `None` | None |
| **Cornelia Street Cafe** | West Village | `not_found` | `None` | [Website](http://www.corneliastreetcafe.com/underground/performances.asp) |
| **El Barrio's Artspace PS109** | East Harlem | `not_found` | `None` | [Website](https://www.artspace.org/ps109) |
| **Erv's on Beekman** | Prospect Lefferts Gardens | `not_found` | `None` | None |
| **Farafina Café & Lounge** | Harlem | `not_found` | `None` | None |
| **Fat Cat** | Greenwich Village | `not_found` | `None` | None |
| **Gin Fizz Harlem** | Harlem | `not_found` | `None` | None |
| **Ginny’s Supper Club** | Harlem | `not_found` | `None` | [Website](https://www.ginnyssupperclub.com/) |
| **Hamilton’s Bar and Kitchen** | Harlem | `not_found` | `None` | None |
| **JAZZ 966** | Clinton Hill | `not_found` | `None` | None |
| **Jazz Standard** | Flatiron | `KovZpZAFIaaA` | `None` | [Website](http://www.jazzstandard.com/) |
| **Jazz at Kitano** | Midtown | `not_found` | `None` | None |
| **Jules Bistro** | East Village | `not_found` | `None` | None |
| **Kitano Hotel** | Midtown Manhattan | `not_found` | `None` | [Website](https://thekitano.com/) |
| **LunÀtico** | Bed-Stuy | `Z7r9jZaAjj` | `None` | [Website](https://www.barlunatico.com/calendar) |
| **Metropolitan Room** | Midtown Manhattan | `not_found` | `None` | None |
| **Minton’s Playhouse** | Harlem | `not_found` | `None` | [Events Link](http://mintonsharlem.com/calendar/) |
| **Paris Blues** | Harlem | `not_found` | `None` | None |
| **Patrick’s Place** | Harlem | `not_found` | `None` | [Events Link](https://www.patricksplaceharlem.com/events) |
| **Red Rooster Harlem** | Harlem | `not_found` | `None` | [Website](http://www.ginnyssupperclub.com/) |
| **Rue B** | East Village | `not_found` | `None` | None |
| **Showmans Jazz Club** | Harlem | `not_found` | `None` | None |
| **Showman’s** | Harlem | `not_found` | `None` | None |
| **Silvana** | Harlem | `not_found` | `None` | [Events Link](http://silvana-nyc.com/) |
| **Smoke** | Upper West Side | `KovZpZAdIkaA` | `None` | [Website](http://www.smokejazz.com/) |
| **St. Mazie** | Williamsburg | `not_found` | `None` | [Events Link](https://www.stmazie.com/events) |
| **The 55 Bar** | Greenwich Village | `not_found` | `None` | None |
| **The 75 Club** | Tribeca | `not_found` | `None` | [Website](https://www.the75clubnyc.com/) |
| **The Cloak Room** | Harlem | `not_found` | `None` | [Website](https://thecloakroom.bar/) |
| **The Django** | Soho | `not_found` | `None` | [Website](http://djangonyc.com/schedule/) |
| **The Jazz Gallery** | Flatiron | `not_found` | `None` | [Events Link](http://jazzgallery.nyc/tickets/index.php?a=697851919) |
| **The Jazz Genius** | Lower East Side | `not_found` | `None` | None |
| **The Jazz Loft** | Stony Brook | `not_found` | `None` | None |
| **The Kitano Hotel Jazz** | Murray Hill | `not_found` | `None` | None |
| **The Porch** | Harlem | `not_found` | `None` | [Events Link](https://www.theporchnyc.com/events.html) |
| **The Wilky** | Bedford-Stuyvesant | `not_found` | `None` | None |
| **Tomi Jazz** | Midtown Manhattan | `not_found` | `None` | [Events Link](https://www.tomijazz.com/events.html) |
| **Williamsburg Music Center** | Williamsburg | `not_found` | `None` | [Events Link](https://www.wmcjazz.org/) |
| **Zinc Bar** | Greenwich Village | `KovZpZAJeaIA` | `None` | [Website](https://zincbar.com/) |

### opera houses (45 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Aaron Davis Hall at The City College of New York** | Harlem | `None` | `None` | [Events Link](https://citycollegecenterforthearts.org/events) |
| **Academy of Music (New York City)** | Manhattan | `None` | `None` | None |
| **Amato Opera** | NoHo | `None` | `None` | None |
| **Astor Opera House** | NoHo | `None` | `None` | None |
| **Bronx Opera Company** | Norwood | `None` | `None` | [Events Link](https://www.bronxopera.org/) |
| **Bronx Opera House** | Bronx | `None` | `None` | None |
| **Century Theatre (Central Park West)** | Upper West Side | `None` | `None` | None |
| **Chelsea Opera** | Hell's Kitchen | `None` | `None` | None |
| **Dicapo Opera Theatre** | Upper East Side | `None` | `None` | None |
| **Gelsey Kirkland Arts Center** | DUMBO | `None` | `None` | None |
| **Gelsey Kirkland ArtsCenter** | Dumbo | `None` | `None` | None |
| **HappyLucky no.1** | Crown Heights | `None` | `None` | None |
| **Harlem Opera House** | Harlem | `None` | `None` | None |
| **Irondale Center** | Fort Greene | `None` | `None` | [Events Link](https://irondale.org/) |
| **Lehman Stages** | Bedford Park | `None` | `None` | [Events Link](https://lehmanstages.org/) |
| **Manhattan Center** | Midtown | `None` | `None` | [Events Link](https://www.mc34.com/) |
| **Metropolitan Opera** | Upper West Side | `None` | `None` | [Events Link](https://www.metopera.org/season/events/) |
| **Metropolitan Opera House (39th Street)** | Midtown | `None` | `None` | None |
| **Metropolitan Opera House (Lincoln Center)** | Upper West Side | `None` | `None` | [Events Link](https://www.metopera.org/season/events/) |
| **NYU Frederick Loewe Theatre** | Greenwich Village | `None` | `None` | None |
| **NYU Provincetown Playhouse** | Greenwich Village | `None` | `None` | None |
| **National Opera Center** | Midtown | `None` | `None` | [Events Link](https://www.operaamerica.org/programs/events/opera-america-salutes/) |
| **New Ohio Theatre** | West Village | `None` | `None` | None |
| **New Stage Theatre Company** | Midtown | `None` | `None` | [Events Link](http://newstagetheatre.org/) |
| **New Workshop Theater** | Midwood | `None` | `None` | None |
| **Newhouse Center for Contemporary Art** | Livingston | `None` | `None` | [Events Link](https://www.snug-harbor.org/newhouse-center-for-contemporary-art/events.html) |
| **Ophelia Theatre Group** | Astoria | `None` | `None` | None |
| **Palmo's Opera House** | Manhattan | `None` | `None` | None |
| **Signature Theatre - The Pershing Square Signature Center** | Hell's Kitchen | `None` | `None` | [Events Link](https://signaturetheatre.org/show/mother-russia/) |
| **Staten Island Playhouse** | St. George | `None` | `None` | None |
| **Staten Island Shakespearean Theatre Company** | Elm Park | `None` | `None` | [Website](https://www.sistny.org/) |
| **The 13th Street Repertory Company** | Greenwich Village | `None` | `None` | None |
| **The Heights Players** | Brooklyn Heights | `None` | `None` | [Events Link](https://www.heightsplayers.org/calendar) |
| **The Hindu Temple Auditorium** | Flushing | `None` | `None` | None |
| **The Juilliard School - Morse Recital Hall** | Lincoln Square | `None` | `None` | [Events Link](https://www.juilliard.edu/campus-life/performance-venues/morse-recital-hall/events.html) |
| **The Little Victory Theatre** | Travis-Chelsea | `None` | `None` | None |
| **The Lovinger Theatre at Lehman College** | Bedford Park | `None` | `None` | None |
| **The Mark O'Donnell Theater at the Entertainment Community Fund Arts Center** | Downtown Brooklyn | `None` | `None` | None |
| **The Mercury Store** | Gowanus | `None` | `None` | [Website](https://mercurystore.com/) |
| **The Mezzanine Theatre at A.R.T./New York Theatres** | Hell's Kitchen | `None` | `None` | None |
| **The Opera Next Door** | Park Slope | `None` | `None` | None |
| **The Peoples Theatre** | Inwood | `None` | `None` | None |
| **The Secret Theatre** | Long Island City | `None` | `None` | [Events Link](https://secrettheatre.com/events) |
| **The Space at Irondale** | Fort Greene | `None` | `None` | [Events Link](https://irondale.org/) |
| **Titan Theatre Company** | Flushing Meadows Corona Park | `None` | `None` | [Events Link](https://www.titantheatrecompany.com/tickets) |

### outdoor concert venues amphitheaters (42 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Al Quiñones Playground Amphitheater** | Longwood | `not_found` | `None` | [Website](https://www.nycgovparks.org/parks/al-quinones-playground) |
| **Ample Hills Creamery at The Fireboat House** | Dumbo | `not_found` | `None` | None |
| **Arlo SoHo (A.R.T. SoHo)** | SoHo | `not_found` | `None` | [Website](https://thearlohotels.com/soho/eat-drink/art-soho/) |
| **Astoria Park Great Lawn** | Astoria | `not_found` | `None` | [Website](https://www.nycgovparks.org/parks/astoria-park) |
| **Brooklyn Army Terminal** | Sunset Park | `Z7r9jZaA7j` | `None` | [Events Link](https://brooklynarmyterminal.com/events.html) |
| **Corporal Allan F. Kivlehan Park** | New Dorp Beach | `not_found` | `None` | [Website](https://www.nycgovparks.org/parks/corporal-allan-f-kivlehan-park) |
| **Father Macris Park** | Graniteville | `not_found` | `None` | [Website](https://www.nycgovparks.org/parks/father-macris-park) |
| **Ford Amphitheater at Coney Island Boardwalk** | Coney Island | `not_found` | `None` | [Website](https://www.coneyislandlive.com/) |
| **Forrest Point** | Bushwick | `not_found` | `None` | None |
| **Fotografiska New York (The Chapel Bar)** | Gramercy Park | `not_found` | `None` | None |
| **Frying Pan** | Chelsea | `not_found` | `None` | [Events Link](https://www.fryingpan.com/) |
| **George Seuffert, Sr. Bandshell** | Woodhaven | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Gil Scott-Heron Amphitheater** | Mott Haven | `not_found` | `None` | None |
| **Hamilton Park** | St. George | `not_found` | `None` | None |
| **Lemon's at The Wythe Hotel** | Williamsburg | `not_found` | `None` | None |
| **Midland Beach Splaza** | Midland Beach | `not_found` | `None` | None |
| **Orchard Beach Pavilion** | Pelham Bay Park | `not_found` | `None` | None |
| **Pier 45** | West Village | `KovZ917AhFJ` | `None` | [Events Link](https://hudsonriverpark.org/locations/pier-45/events.html) |
| **Radio Park at Rockefeller Center** | Midtown | `not_found` | `None` | [Events Link](https://www.rockefellercenter.com/amenities/radio-park/events.html) |
| **Rockaway Beach Amphitheater** | Rockaway Beach | `not_found` | `None` | None |
| **Rockaway Hotel Rooftop** | Rockaway Park | `not_found` | `None` | [Website](https://rockawayhotel.com/dine-drink/the-rooftop) |
| **Soundview Park Amphitheater** | Soundview | `not_found` | `None` | None |
| **South Beach Boardwalk** | South Beach | `not_found` | `None` | [Website](https://www.nycgovparks.org/parks/franklin-d-roosevelt-boardwalk-and-beach) |
| **Southpoint Park on Roosevelt Island** | Roosevelt Island | `not_found` | `None` | None |
| **The Bronx Brewery & Empanology** | Port Morris | `not_found` | `None` | [Website](https://thebronxbrewery.com/) |
| **The Cantor Roof Garden Bar at The Met** | Upper East Side | `not_found` | `None` | None |
| **The Good Roof at The Good Fork** | Red Hook | `not_found` | `None` | None |
| **The Greens at Pier 17** | Seaport | `not_found` | `None` | None |
| **The Oasis at The William Vale** | Williamsburg | `not_found` | `None` | None |
| **The Public Square and Gardens at Hudson Yards** | Hudson Yards | `Z7r9jZaAka` | `None` | None |
| **The Roof at Park South** | Rose Hill | `not_found` | `None` | None |
| **The Roof at Superior Ingredients** | Williamsburg | `not_found` | `None` | [Website](https://www.superioringredients.com/) |
| **The Roof at Whole Foods Market** | Gowanus | `not_found` | `None` | None |
| **The Rooftop at The Hoxton, Williamsburg** | Williamsburg | `not_found` | `None` | [Events Link](https://www.laserwolfbrooklyn.com/events) |
| **The Rooftop at the Box House Hotel** | Greenpoint | `not_found` | `None` | [Events Link](https://www.theboxhousehotel.com/rooftop/events) |
| **The Ruins at Knockdown Center** | Maspeth | `not_found` | `None` | [Events Link](https://knockdown.center/) |
| **The Unisphere at Flushing Meadows Corona Park** | Flushing | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **The William Vale Pool + Terrace** | Williamsburg | `not_found` | `None` | [Events Link](https://www.thewilliamvale.com/wp-json/tribe/events/v1/) |
| **Valentino Pier** | Red Hook | `not_found` | `None` | None |
| **Waterline Square Park** | Upper West Side | `not_found` | `None` | [Website](https://www.waterlinesquare.com/park/) |
| **Westerleigh Park Gazebo** | Westerleigh | `not_found` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Westlight at The William Vale** | Williamsburg | `not_found` | `None` | [Website](https://www.westlightnyc.com/) |

### off-broadway theaters (41 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **59E59 Theaters** | Midtown East | `ZFr9jZa7Fv` | `None` | [Website](http://www.59e59.org/spaces.php) |
| **AMT Theater** | Hell's Kitchen | `Z7r9jZaegg` | `None` | [Website](https://www.amttheater.org/rentals) |
| **Asylum NYC** | Flatiron | `Z7r9jZa7My` | `None` | None |
| **Barrow Street Theatre** | West Village | `not_found` | `None` | None |
| **Baryshnikov Arts Center** | Hell's Kitchen | `ZFr9jZ7FAa` | `None` | [Website](https://bacnyc.org/rent) |
| **Chain Theatre** | Midtown | `Z7r9jZaAK1` | `None` | [Website](https://www.chaintheatre.org/space-rental) |
| **Connelly Theater** | East Village | `KovZ917ARm0` | `None` | [Website](https://www.connellytheater.org/) |
| **DR2 Theatre** | Union Square | `not_found` | `None` | [Events Link](http://www.darylroththeatre.com/rentals/dr2-theatre/) |
| **Gene Frankel Theatre** | NoHo | `KovZpZAFIJtA` | `None` | [Website](http://www.genefrankeltheatre.com/theater-rental.html) |
| **Greenwich House Theatre** | Greenwich Village | `not_found` | `None` | None |
| **Jerry Orbach Theater** | Midtown | `Z7r9jZadyf` | `None` | [Website](https://www.thetheatercenter.com/jerry-orbach-theater) |
| **Laura Pels Theatre** | Theater District | `ZFr9jZ7A1d` | `None` | None |
| **Minetta Lane Theatre** | Greenwich Village | `ZFr9jZeevd` | `None` | [Website](http://minettalanenyc.com/) |
| **Orpheum Theater** | East Village | `not_found` | `None` | None |
| **Playhouse 46 at St. Luke’s** | Hell's Kitchen | `not_found` | `None` | [Events Link](https://playhouse46.org/rental/events.html) |
| **Playwrights Horizons** | Hell's Kitchen | `ZFr9jZAvdv` | `None` | [Website](https://www.playwrightshorizons.org/space-rental/) |
| **Primary Stages** | Midtown East | `ZFr9jZ77d6` | `None` | [Events Link](https://primarystages.org/shows-events) |
| **Primary Stages at 59E59 Theaters** | Midtown East | `not_found` | `None` | [Events Link](https://primarystages.org/shows-events) |
| **Puerto Rican Traveling Theatre** | Hell's Kitchen | `KovZpZAJAvvA` | `None` | [Website](http://pregonesprtt.org/you/rentals/) |
| **Rattlestick Playwrights Theater** | Greenwich Village | `Z7r9jZaeVK` | `None` | [Website](https://www.rattlestick.org/rent-our-space) |
| **Signature Theatre Company** | Hell's Kitchen | `not_found` | `None` | [Events Link](https://signaturetheatre.org/show/mother-russia/) |
| **Soho Playhouse** | Soho | `KovZpZAIv77A` | `None` | [Website](http://www.sohoplayhouse.com/rental-info) |
| **Soho Repertory Theatre** | TriBeCa | `not_found` | `None` | [Events Link](https://sohorep.org/shows/watch-me-walk/) |
| **St. Luke's Theatre** | Theater District | `ZFr9jZae77` | `None` | None |
| **Steve & Marie Sgouros Theatre** | Greenwich Village | `not_found` | `None` | [Website](http://playerstheatre.com/steve-and-marie-sgouros-theatre.html) |
| **The Acorn Theatre** | Hell's Kitchen | `ZFr9jZ77kA` | `None` | None |
| **The American Place Theatre** | Hell's Kitchen | `not_found` | `None` | None |
| **The Clemente Soto Vélez Cultural & Educational Center** | Lower East Side | `not_found` | `None` | [Website](https://clemente.org/) |
| **The Flea Theater** | TriBeCa | `not_found` | `None` | [Events Link](https://theflea.org/programs-1) |
| **The Gym at Judson** | Greenwich Village | `ZFr9jZa7Fd` | `None` | [Website](https://www.thegymatjudson.com/contact) |
| **The Lion Theatre** | Hell's Kitchen | `ZFr9jZae7e` | `None` | None |
| **The Players Theatre** | Greenwich Village | `Z7r9jZadV6` | `None` | [Website](http://theplayerstheater.com/) |
| **The Riverside Theatre** | Morningside Heights | `not_found` | `None` | None |
| **The Theatre at St. Jean’s** | Upper East Side | `not_found` | `None` | [Events Link](https://www.sjbny.org/the-theatre-st-jean/events) |
| **Theater 555** | Hell's Kitchen | `Z7r9jZaA1J` | `None` | [Website](https://www.theater555.com/rental-rates) |
| **Theatre 80** | East Village | `KovZpZA1EkJA` | `None` | [Website](https://theatre80.wordpress.com/) |
| **Theatre Row** | Hell's Kitchen | `Z7r9jZa7Tw` | `None` | [Website](https://bfany.org/theatre-row/theater-rentals/) |
| **Theatre at St. Clement’s** | Hell's Kitchen | `KovZ917AYEB` | `None` | [Website](http://www.stclementsnyc.org/theatre.html) |
| **Urban Stages** | Chelsea | `not_found` | `None` | [Events Link](https://www.urbanstages.org/events) |
| **Vineyard Theatre** | Union Square | `ZFr9jZA7Fe` | `None` | [Website](https://vineyardtheatre.org/rent-us-out/) |
| **WP Theater** | Upper West Side | `not_found` | `None` | [Events Link](https://wptheater.org/tickets/) |

### outdoor event spaces gardens (35 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **1100 Block Bergen Street Community Garden** | Crown Heights | `None` | `None` | None |
| **61 Franklin Street Garden** | Greenpoint | `None` | `None` | None |
| **6BC Botanical Garden** | East Village | `None` | `None` | [Website](http://6bc.org/) |
| **Above at the Hilton Garden Inn Staten Island** | Bloomfield | `None` | `None` | None |
| **Antun's Montauk Gardens** | Queens Village | `None` | `None` | None |
| **Battery Gardens** | Financial District | `None` | `None` | None |
| **Campos Community Garden** | East Village | `None` | `None` | None |
| **Clinton Community Garden** | Hell's Kitchen | `None` | `None` | [Events Link](https://clintoncommunitygarden.org/event-free-fire-winterlands-sensasi-battle-royale-di-dunia-bersalju/) |
| **Cooper Hewitt, Smithsonian Design Museum - Arthur Ross Terrace and Garden** | Upper East Side | `None` | `None` | None |
| **Creative Little Garden** | East Village | `None` | `None` | [Events Link](http://www.creativelittlegarden.org/events) |
| **El Puente Espíritu Tierra Community Garden** | Williamsburg | `None` | `None` | [Website](https://elpuente.us/) |
| **Gitano Garden of Love** | Hudson Square | `None` | `None` | None |
| **Green Oasis Community Garden** | East Village | `None` | `None` | None |
| **Greene Acres Community Garden** | Bedford-Stuyvesant | `None` | `None` | None |
| **Harlem Roots Community Garden** | Harlem | `None` | `None` | None |
| **Joe Holzka Community Garden** | West New Brighton | `None` | `None` | None |
| **Jue Lan Club** | Flatiron District | `None` | `None` | None |
| **Liz Christy Community Garden** | Bowery | `None` | `None` | None |
| **Majestic Gardens** | Rocky Point | `None` | `None` | None |
| **Northside Community Garden** | Williamsburg | `None` | `None` | None |
| **Parish Hall** | Richmond Hill | `None` | `None` | None |
| **Red Hook Art Gallery on the Water (New Earth Museum)** | Red Hook | `None` | `None` | None |
| **Saint Nicholas Miracle Garden** | Harlem | `None` | `None` | None |
| **Skyline Community Garden** | New Brighton | `None` | `None` | None |
| **Sunnyside Community Garden** | Sunnyside | `None` | `None` | None |
| **Target Bronx Community Garden** | Highbridge | `None` | `None` | [Events Link](https://www.nyrp.org/en/gardens/target-bronx-community-garden/events.html) |
| **The Gardens of St. Nicholas Park** | Harlem | `None` | `None` | [Website](https://www.stnicholaspark.org/) |
| **The Morgan Library & Museum - Gilbert Court** | Murray Hill | `None` | `None` | [Events Link](https://www.themorgan.org/drawing-institute/events) |
| **The Palm House at the Brooklyn Botanic Garden** | Prospect Park | `None` | `None` | None |
| **The Pine Hollow Club** | East Norwich | `None` | `None` | [Website](https://www.pinehollowclub.com/) |
| **The Vessel (Public Square and Gardens)** | Hudson Yards | `None` | `None` | [Events Link](https://www.hudsonyardsnewyork.com/events/celebrate-lunar-new-year-hudson-yards) |
| **Two Coves Community Garden** | Astoria | `None` | `None` | None |
| **Victory Garden Cafe** | Astoria | `None` | `None` | [Website](https://victorygardennyc.com/) |
| **Victory Garden Café** | Astoria | `None` | `None` | None |
| **Westervelt Family and Community Garden** | New Brighton | `None` | `None` | None |

### dance venues ballet contemporary (33 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **92nd Street Y, Harkness Dance Center** | Upper East Side | `not_found` | `None` | None |
| **Alvin Ailey American Dance Theater** | Hell's Kitchen | `not_found` | `None` | [Events Link](https://www.alvinailey.org/events/ailey-atlanta-ga-21326) |
| **Alvin Ailey American Dance Theater at the Joan Weill Center for Dance** | Midtown | `not_found` | `None` | [Events Link](https://www.alvinailey.org/calendar) |
| **American Chamber Ballet** | NYC | `not_found` | `None` | None |
| **American Negro Ballet Company** | NYC | `not_found` | `None` | None |
| **Ballet Deviare** | NYC | `not_found` | `None` | [Website](https://www.balletdeviare.org/) |
| **Ballez** | NYC | `not_found` | `None` | [Events Link](https://www.ballez.org/shows) |
| **Cedar Lake Contemporary Ballet** | Chelsea | `not_found` | `None` | None |
| **City Center** | Midtown | `KovZpaplHe` | `None` | [Website](http://www.nycitycenter.org/Home) |
| **Complexions Contemporary Ballet** | NYC | `not_found` | `None` | None |
| **Cotton Club Boys** | Harlem | `not_found` | `None` | None |
| **D Underbelly** | NYC | `not_found` | `None` | None |
| **Dance Theatre of Harlem** | Harlem | `not_found` | `None` | [Website](http://www.dancetheatreofharlem.com/) |
| **DanceBrazil** | NYC | `not_found` | `None` | None |
| **Eryc Taylor Dance** | NYC | `not_found` | `None` | [Events Link](https://etd.nyc/calendar) |
| **Grand Union** | NYC | `not_found` | `None` | None |
| **Harkness Ballet** | Upper East Side | `not_found` | `None` | None |
| **ILuminate** | NYC | `not_found` | `None` | [Events Link](https://www.iluminate.com/events) |
| **Ice Theatre of New York** | NYC | `not_found` | `None` | [Events Link](https://www.icetheatre.org/tickets.html) |
| **Lar Lubovitch Dance Company** | NYC | `not_found` | `None` | [Events Link](https://www.lubovitch.org/events) |
| **Lydia Johnson Dance** | NYC | `not_found` | `None` | [Events Link](https://www.lydiajohnsondance.org/events-recent) |
| **New Dance Group** | NYC | `not_found` | `None` | None |
| **New York City Center** | Midtown West | `KovZpaplHe` | `None` | [Website](https://www.nycitycenter.org/events-tickets/2024-2025-season/) |
| **New York Negro Ballet** | NYC | `not_found` | `None` | None |
| **PMT Dance Studio** | Flatiron District | `not_found` | `None` | None |
| **PearsonWidrig DanceTheater** | NYC | `not_found` | `None` | [Events Link](https://www.pearsonwidrig.org/calendar) |
| **Peggy Spina Tap Company** | SoHo | `not_found` | `None` | None |
| **Peridance Center** | East Village | `not_found` | `None` | [Events Link](https://www.peridance.com/) |
| **Reggie Wilson/Fist and Heel Performance Group** | Brooklyn | `not_found` | `None` | [Events Link](https://www.fistandheelperformancegroup.org/) |
| **The Dazzle Dancers** | NYC | `not_found` | `None` | None |
| **The Rockettes** | Midtown | `not_found` | `None` | [Website](https://www.rockettes.com/) |
| **White Oak Dance Project** | NYC | `not_found` | `None` | None |
| **Whitey's Lindy Hoppers** | Harlem | `not_found` | `None` | None |

### NYC parks with free events (32 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Astoria Park** | Astoria | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Betsy Head Park** | Brownsville | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Bloomingdale Park** | Woodrow | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Bowne Park** | Flushing | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Butterfly Garden** | South Slope | `None` | `None` | [Events Link](https://bronxzoo.com/tickets) |
| **Concrete Plant Park** | Foxhurst | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Ewen Park** | Riverdale | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Henry Hudson Park** | Spuyten Duyvil | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Highbridge Park** | Washington Heights | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Highland Park** | Cypress Hills | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Julio Carballo Fields** | Hunts Point | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Juniper Valley Park** | Middle Village | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Kissena Park** | Kissena Park | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Leif Ericson Park** | Bay Ridge | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Marcus Garvey Park** | Harlem, Manhattan | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Marine Park** | Marine Park | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Marine Park - Salt Marsh Nature Center** | Marine Park | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Morningside Park** | Morningside Heights, Manhattan | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Northerleigh Park** | Port Richmond | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Owl's Head Park** | Bay Ridge | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Pugsley Creek Park** | Clason Point | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Queensbridge Park** | Long Island City | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Seton Falls Park** | Edenwald | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Soundview Park** | Soundview | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **St. James Park** | Fordham | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **St. John's Recreation Center** | Crown Heights | `None` | `None` | [Website](https://www.nycgovparks.org/facilities/recreationcenters/B082) |
| **Sunset Park** | Sunset Park | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Tompkins Square Park** | East Village, Manhattan | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Torsney Playground** | Sunnyside | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Williamsbridge Oval** | Norwood | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Willowbrook Park** | Willowbrook | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Wolfe's Pond Park** | Prince's Bay | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |

### outdoor movies parks NYC (31 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **6 & B Garden** | East Village | `None` | `None` | None |
| **A.R.R.O.W. Field House** | Astoria | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Abraham Lincoln Playground** | Harlem | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **All People's Community Garden** | Bedford-Stuyvesant | `None` | `None` | None |
| **Bedford-Stuyvesant Community Garden** | Bedford-Stuyvesant | `None` | `None` | None |
| **Bensonhurst Park** | Bensonhurst | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Crocheron Park** | Bayside | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Dias y Flores Community Garden** | East Village | `None` | `None` | None |
| **Down to Earth Garden** | East Village | `None` | `None` | [Events Link](https://downtoearthgarden.org/category/external-garden-related-events/) |
| **Elizabeth Street Garden Movie Nights** | Nolita, Manhattan | `None` | `None` | [Events Link](https://www.elizabethstreetgarden.com/calendar) |
| **First Street Garden** | East Village | `None` | `None` | None |
| **Garden of Angels** | Bedford-Stuyvesant | `None` | `None` | None |
| **Greene Garden Movie Nights** | Fort Greene | `None` | `None` | None |
| **Hamilton Fish Park** | Lower East Side | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Human Compass Garden** | Columbia Street Waterfront District | `None` | `None` | None |
| **Intrepid Museum's Summer Movie Series** | Hell's Kitchen, Manhattan | `None` | `None` | None |
| **J. Hood Wright Park** | Washington Heights | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Kenkeleba House Garden** | East Village | `None` | `None` | None |
| **Lincoln Terrace / Arthur S. Somers Park** | Crown Heights | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Maria Hernandez Park** | Bushwick | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Narrows Botanical Gardens** | Bay Ridge | `None` | `None` | None |
| **Nostrand Playground** | East Flatbush | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Pirate's Cove Garden** | Columbia Street Waterfront District | `None` | `None` | None |
| **Queens Drive-In at The New York Hall of Science** | Flushing Meadows Corona Park | `None` | `None` | None |
| **Shore Park and Parkway** | Bay Ridge | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Sorrentino Recreation Center** | Far Rockaway | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **South Brooklyn Children's Garden** | Columbia Street Waterfront District | `None` | `None` | [Website](https://www.southbrooklynchildrensgarden.org/) |
| **Summit Street Community Garden** | Carroll Gardens | `None` | `None` | None |
| **Sun, Wind and Shade Oasis Garden** | Morrisania | `None` | `None` | None |
| **The Amazing Garden** | Columbia Street Waterfront District | `None` | `None` | None |
| **The Backyard Community Garden** | Columbia Street Waterfront District | `None` | `None` | None |

### history museums (31 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Alice Austen House Museum** | Rosebank | `None` | `None` | [Events Link](https://aliceausten.org/events/exhibition-opening-resilient-communities/) |
| **American Irish Historical Society** | Upper East Side | `None` | `None` | None |
| **Bartow-Pell Mansion Museum** | Pelham Bay Park | `None` | `None` | [Events Link](https://www.bartowpellmansionmuseum.org/events-2/) |
| **Douglaston/Little Neck Historical Society** | Douglaston | `None` | `None` | None |
| **Fort Wadsworth Visitor Center** | Fort Wadsworth | `None` | `None` | None |
| **Garibaldi-Meucci Museum** | Rosebank | `None` | `None` | None |
| **Gracie Mansion** | Yorkville | `None` | `None` | [Events Link](https://www.graciemansion.org/exhibitions) |
| **Greater Astoria Historical Society** | Astoria | `None` | `None` | [Events Link](https://astorialic.org/) |
| **Green-Wood Cemetery** | Greenwood Heights | `None` | `None` | [Website](https://www.green-wood.com/) |
| **Harbor Defense Museum** | Fort Hamilton | `None` | `None` | None |
| **Huntington Free Library and Reading Room** | Westchester Square | `None` | `None` | [Events Link](https://huntingtonfreelibrary.org/events/) |
| **Italian American Museum** | Little Italy | `None` | `None` | [Events Link](https://www.italianamericanmuseum.org/museum-events/) |
| **Jacques Marchais Museum of Tibetan Art** | Lighthouse Hill | `None` | `None` | [Events Link](https://www.tibetanmuseum.org) |
| **King Caesar's House** | Duxbury | `None` | `None` | None |
| **Lefferts Historic House** | Prospect Lefferts Gardens | `None` | `None` | [Events Link](https://www.prospectpark.org/visit/places-to-go/lefferts-historic-house/events.html) |
| **Lent-Riker-Smith Homestead** | East Elmhurst | `None` | `None` | None |
| **Leo Baeck Institute - New York** | Union Square | `None` | `None` | [Events Link](https://www.lbi.org/events/) |
| **Lower Manhattan Historical Association** | Financial District | `None` | `None` | None |
| **Mount Vernon Hotel Museum & Garden** | Upper East Side | `None` | `None` | None |
| **Museum of Maritime Navigation and Communication** | Stapleton | `None` | `None` | None |
| **New York City Police Museum** | Financial District | `None` | `None` | None |
| **Newtown Historical Society** | Elmhurst | `None` | `None` | None |
| **Preservation League of Staten Island** | West New Brighton | `None` | `None` | None |
| **Richmond Hill Historical Society** | Richmond Hill | `None` | `None` | [Events Link](https://www.richmondhillhistory.org/events) |
| **Rubin Museum of Art** | Chelsea | `None` | `None` | [Events Link](https://rubinmuseum.org/projects-exhibitions/) |
| **Sandy Ground Historical Society** | Rossville | `None` | `None` | None |
| **Seguine Mansion** | Prince's Bay | `None` | `None` | None |
| **The Olde Towne of Flushing Burial Ground** | Flushing | `None` | `None` | None |
| **Tottenville Historical Society** | Tottenville | `None` | `None` | None |
| **Williamsburg Art & Historical Center** | Williamsburg | `None` | `None` | [Events Link](https://www.wahcenter.net/exhibitions-and-events/) |
| **Woodlawn Cemetery** | Woodlawn | `None` | `None` | [Events Link](https://www.thewoodlawncemetery.org/conservancy/tours-events/) |

### science museums (29 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Blue Heron Park Nature Center** | Annadale | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Bronx River Alliance** | Bronx | `None` | `None` | [Events Link](https://bronxriver.org/category/events) |
| **Brooklyn College Geology Museum** | Flatbush | `None` | `None` | [Events Link](https://brooklyn.edu/geology-museum/events.html) |
| **Charles A. Dana Discovery Center** | Central Park | `None` | `None` | [Events Link](https://www.centralparknyc.org/activities/events) |
| **City Reliquary Museum** | Williamsburg | `None` | `None` | [Events Link](https://www.cityreliquary.org/exhibition/exhibitionhall/) |
| **City Tech Science Fiction Collection** | Downtown Brooklyn | `None` | `None` | [Website](https://library.citytech.cuny.edu/collections/scienceFiction) |
| **Earth Matter NY** | Governors Island | `None` | `None` | [Events Link](https://earthmatter.org/participate/adult-apprenticeship-program/) |
| **Genovesi Environmental Study Center** | Bergen Beach | `None` | `None` | None |
| **Harlem Grown** | Harlem | `None` | `None` | [Events Link](https://www.harlemgrown.org/community-events) |
| **High Rock Park Nature Center** | Egbertville | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Holographic Studios** | Kips Bay | `None` | `None` | [Events Link](https://www.holographer.com/) |
| **Idlewild Park Preserve Environmental Science Learning Center** | Springfield Gardens | `None` | `None` | [Website](https://www.easternqueensalliance.org/) |
| **Inwood Hill Nature Center** | Inwood | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Inwood Hill Park Nature Center** | Inwood | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Jamaica Bay Wildlife Refuge** | Broad Channel | `None` | `None` | None |
| **Jamaica Bay Wildlife Refuge Visitor Center** | Broad Channel | `None` | `None` | None |
| **Kew Kids Forest School** | Kew Gardens | `None` | `None` | [Events Link](https://www.kewkids.com/events) |
| **Micro Museum** | Boerum Hill | `None` | `None` | None |
| **Mmuseumm** | Tribeca | `None` | `None` | [Website](http://www.mmuseumm.com/) |
| **Pelham Bay Nature Center** | Pelham Bay Park | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Prospect Park Audubon Center** | Prospect Park | `None` | `None` | [Events Link](https://www.prospectpark.org/visit/places-to-go/audubon-center/events.html) |
| **Queens Landing Boathouse and Environmental Education Center** | Long Island City | `None` | `None` | None |
| **Socrates Sculpture Park** | Long Island City | `None` | `None` | [Events Link](https://socratessculpturepark.org/program/2025-socrates-gala/) |
| **Solar One** | Kips Bay | `None` | `None` | None |
| **Solar One Environmental Education Center** | Kips Bay | `None` | `None` | None |
| **The Burns Archive** | Murray Hill | `None` | `None` | [Events Link](https://www.burnsarchive.com/exhibitions) |
| **Urban Tree House** | Governors Island | `None` | `None` | None |
| **Van Cortlandt Nature Center** | Van Cortlandt Park | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Van Cortlandt Park Nature Center** | Fieldston | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |

### cultural centers (27 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Afro Latin Jazz Alliance** | Harlem | `None` | `None` | [Events Link](https://afrolatinjazz.org/events/belongo-at-the-harlem-fine-arts-show) |
| **Arab-American Family Support Center** | Brooklyn Heights | `None` | `None` | [Website](https://www.aafscny.org/) |
| **Asian American Arts Centre** | Lower East Side | `None` | `None` | None |
| **Black Spectrum Theatre Co.** | St. Albans | `None` | `None` | [Events Link](https://www.blackspectrum.com/) |
| **Bronx Jewish Center** | Pelham Parkway | `None` | `None` | [Events Link](https://www.bronxjewishcenter.org/bjccalendar) |
| **Carlos Lezama Archives and Caribbean Cultural Center** | Crown Heights | `None` | `None` | None |
| **Center for Family Life in Sunset Park** | Sunset Park | `None` | `None` | [Events Link](https://sco.org/programs/) |
| **Center for Italian Modern Art** | SoHo | `None` | `None` | None |
| **Cultural Museum of African Art** | Bedford–Stuyvesant | `None` | `None` | None |
| **DCTV (Downtown Community Television Center)** | Chinatown | `None` | `None` | None |
| **Davidson Cornerstone Community Center** | Morrisania | `None` | `None` | None |
| **Julia De Burgos Latino Cultural Center** | East Harlem | `None` | `None` | None |
| **King Juan Carlos I of Spain Center at NYU** | Greenwich Village | `None` | `None` | [Events Link](https://www.kjcc.org/events/) |
| **Mind-Builders Creative Arts Center** | Williamsbridge | `None` | `None` | [Events Link](https://www.mind-builders.org/programs/) |
| **Museum of Contemporary African Diasporan Arts** | Fort Greene | `None` | `None` | [Events Link](https://mocada.org/events/shani-crowe-red-black-green-exhibition/) |
| **Museum of Contemporary African Diasporan Arts (MoCADA)** | Fort Greene | `None` | `None` | [Events Link](https://mocada.org/events/mocada-radio-live-feb-14-2026/) |
| **New Settlement Community Center** | Mount Eden | `None` | `None` | [Events Link](https://newsettlement.org/events/) |
| **New York Botanical Garden** | Bronx | `None` | `None` | [Events Link](https://www.nybg.org/gardens/bronx-green-up/events/) |
| **Polish & Slavic Center** | Greenpoint | `None` | `None` | [Events Link](https://polishslaviccenter.org/events) |
| **Renaissance Youth Center** | Morrisania | `None` | `None` | [Website](https://www.renaissanceyouth.org/) |
| **Sandy Ground Historical Museum** | South Shore | `None` | `None` | [Events Link](http://sandygroundmuseum.org/) |
| **Shirley Chisholm Recreation Center** | East Flatbush | `None` | `None` | None |
| **Skylight Center** | St. George | `None` | `None` | [Website](https://fountainhouse.org/) |
| **Sotomayor Cornerstone Community Center** | Soundview | `None` | `None` | None |
| **Soundview Cornerstone Community Center** | Soundview | `None` | `None` | None |
| **The Harbor Lights Theater Company** | St. George | `None` | `None` | None |
| **Theatre for the New City** | East Village | `None` | `None` | None |

### best live music venues (26 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **About Last Night** | Bed-Stuy | `not_found` | `None` | None |
| **Arcadia Bar and Kitchen** | Astoria | `not_found` | `None` | None |
| **Astoria Tavern** | Astoria | `not_found` | `None` | [Website](http://astoriatavern.com/) |
| **Brooklyn Music Kitchen** | Fort Greene | `rZ7HnEZ174OAI` | `None` | None |
| **Brooklyn Record Exchange** | Bushwick | `not_found` | `None` | [Website](https://brooklynrecordexchange.com/) |
| **Bunna Cafe** | Bushwick | `not_found` | `None` | [Website](https://bunnacafe.com/) |
| **Caffeine Underground** | Bushwick | `not_found` | `None` | None |
| **Crossroads Cafe** | Bushwick | `not_found` | `None` | None |
| **Damballa** | Bed-Stuy | `not_found` | `None` | None |
| **Dizzy's Club Coca-Cola** | Upper West Side | `not_found` | `None` | [Events Link](https://jazz.org/calendar) |
| **Downtown Music Gallery** | Two Bridges | `not_found` | `None` | [Events Link](https://www.downtownmusicgallery.com/shows.php) |
| **Gaia NoMaya** | Prospect Lefferts Gardens | `not_found` | `None` | [Events Link](https://www.gaianomaya.com/events) |
| **Highside Workshop** | Bushwick | `not_found` | `None` | None |
| **Human Head Records** | East Williamsburg | `not_found` | `None` | None |
| **Irish Whiskey Bar** | Astoria | `not_found` | `None` | None |
| **Jazz at Lincoln Center** | Columbus Circle | `not_found` | `None` | [Events Link](https://jazz.org/calendar) |
| **Maggie Hall's** | Astoria | `not_found` | `None` | [Website](https://www.maggiehalls.com/) |
| **Pallisades** | Bushwick | `not_found` | `None` | None |
| **Peaches** | Bed-Stuy | `not_found` | `None` | [Events Link](https://peachesbrooklyn.com/) |
| **Rough Trade NYC** | Midtown | `KovZpZAJAtlA` | `None` | [Events Link](https://www.roughtrade.com/events.html) |
| **Subrosa** | Meatpacking District | `not_found` | `None` | None |
| **Sundown** | Ridgewood | `not_found` | `None` | None |
| **The Record Shop** | Red Hook | `not_found` | `None` | None |
| **Troost** | Greenpoint | `not_found` | `None` | None |
| **Zürcher Gallery** | NoHo | `not_found` | `None` | [Events Link](https://www.galeriezurcher.com/current-exhibitions) |
| **pinkFROG Cafe** | Williamsburg | `not_found` | `None` | [Events Link](https://www.pinkfrogcafe.com/) |

### parks with programming activities (24 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Baisley Pond Park** | Jamaica | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Blood Root Valley** | Richmondtown | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Blue Heron Park** | Annadale | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Bronx Park** | Bronx Park | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Brook Park Community Garden** | Mott Haven | `None` | `None` | [Website](https://brookpark.org/) |
| **City Hall Park** | Financial District | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **CityParks Junior Golf Center** | Bay Ridge | `None` | `None` | None |
| **College Avenue Garden** | Morrisania | `None` | `None` | None |
| **Eastchester Gardens Community Center** | Eastchester | `None` | `None` | None |
| **Garden of Eden** | Mount Hope | `None` | `None` | None |
| **Greenbelt Native Plant Center** | Travis - Chelsea | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Idlewild Park** | Springfield Gardens | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Jackson Forest Community Garden** | Morrisania | `None` | `None` | None |
| **John Golden Park** | Bayside | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Lincoln Terrace Park** | Crown Heights | `None` | `None` | [Website](https://www.nycgovparks.org/parks/lincoln-terrace-park) |
| **Lost Battalion Hall Recreation Center** | Rego Park | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Metropolitan Recreation Center** | Williamsburg | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Morning Glory Community Garden** | Crotona Park East | `None` | `None` | None |
| **O'Donohue Park Amphitheater** | Far Rockaway | `None` | `None` | None |
| **Poe Park Visitor Center** | Fordham | `None` | `None` | [Website](https://www.nycgovparks.org/facilities/visitorcenters/10) |
| **Rainbow Garden of Life and Health** | Melrose | `None` | `None` | None |
| **River Garden** | West Farms | `None` | `None` | None |
| **Rochdale Park** | Rochdale | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Wishing Well Community Garden** | Morrisania | `None` | `None` | None |

### art museums (23 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **A.I.R. Gallery** | Dumbo | `None` | `None` | [Events Link](https://www.airgallery.org/events-1/unforgettables-2026) |
| **American Academy of Arts and Letters** | Washington Heights | `None` | `None` | [Events Link](https://artsandletters.org/exhibitions) |
| **Americas Society** | Upper East Side | `None` | `None` | [Events Link](https://www.as-coa.org/events/amazonia-acu-conference) |
| **Bartow-Pell Mansion** | Pelham Bay | `None` | `None` | [Events Link](https://www.bartowpellmansionmuseum.org/events-2/) |
| **Bernard Museum of Judaica** | Upper East Side | `None` | `None` | [Events Link](https://www.emanuelnyc.org/about-us/bernard-museum/exhibitions/) |
| **Billiou-Stillwell-Perine House** | Old Town | `None` | `None` | None |
| **Bone Museum** | Williamsburg | `None` | `None` | [Events Link](https://bonemuseum.com/) |
| **Bronx River Art Center** | West Farms | `None` | `None` | [Events Link](http://www.bronxriverart.org/events) |
| **Brooklyn Museum** | Crown Heights | `None` | `None` | [Events Link](https://www.brooklynmuseum.org/programs/poetry-workshop-february-2026/02-22-2026) |
| **Brooklyn Navy Yard Center at BLDG 92** | Brooklyn Navy Yard | `None` | `None` | [Events Link](http://bldg92.org/events) |
| **Center for Art and Culture of Bedford-Stuyvesant** | Bedford–Stuyvesant | `None` | `None` | [Website](https://web.archive.org/web/20160314104630/http://restorationplaza.org/arts-and-culture) |
| **Center for Brooklyn History** | Brooklyn Heights | `None` | `None` | [Events Link](https://www.bklynlibrary.org/exhibitions) |
| **City Reliquary** | Williamsburg | `None` | `None` | [Events Link](https://cityreliquary.org/) |
| **Clemente Soto Velez Cultural and Educational Center** | Lower Manhattan | `None` | `None` | [Website](https://clemente.org/) |
| **Conference House** | Tottenville | `None` | `None` | [Events Link](https://conferencehouse.org/events/) |
| **Cooper Union Galleries** | East Village | `None` | `None` | None |
| **Czech Centre New York** | Upper East Side | `None` | `None` | [Events Link](http://new-york.czechcentres.cz/events.html) |
| **MoMA** | Midtown West | `None` | `None` | [Events Link](https://www.moma.org/calendar/exhibitions/history) |
| **National Museum of the American Indian** | Lower Manhattan | `None` | `None` | [Events Link](https://americanindian.si.edu/events/?trumbaEmbed=view%3Dseries%26seriesid%3D1875643) |
| **New York Transit Museum** | Brooklyn Heights | `None` | `None` | [Events Link](https://www.nytransitmuseum.org/wp-json/tribe/events/v1/) |
| **Queens Museum** | Queens | `None` | `None` | [Events Link](https://queensmuseum.org/whats-on/) |
| **The Met Cloisters** | Washington Heights | `None` | `None` | [Events Link](https://www.metmuseum.org/visit/plan-your-visit/met-cloisters) |
| **The Museum of Broadway** | Times Square/Theatre District | `None` | `None` | [Events Link](https://www.themuseumofbroadway.com/) |

### comedy clubs standup (23 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Arcadia Bar & Kitchen** | Astoria | `not_found` | `None` | None |
| **Arrogant Swine** | East Williamsburg | `not_found` | `None` | None |
| **BKLYN Comedy Club** | Bushwick | `not_found` | `None` | [Website](https://www.bklyncomedyclub.com/) |
| **Bar Lubitsch** | Williamsburg | `not_found` | `None` | None |
| **Cozy Art Land** | Long Island City | `not_found` | `None` | [Website](https://cozyartland.com/) |
| **Fat Black Pussycat** | Greenwich Village | `not_found` | `api` | [Events Link](https://www.comedycellar.com/new-york-line-up/) |
| **Gotham Comedy Club** | Chelsea | `ZFr9jZ6Fk7` | `None` | [Website](https://www.gothamcomedyclub.com/) |
| **Greenpoint Comedy Club** | Greenpoint | `not_found` | `None` | [Events Link](https://www.greenpointcomedy.com/) |
| **Katch Astoria** | Astoria | `not_found` | `None` | [Events Link](https://www.katchastoria.com/events.html) |
| **Nicky's Unisex** | Williamsburg | `not_found` | `None` | None |
| **Park Slope Comedy Spot** | Park Slope | `not_found` | `None` | None |
| **Precious Metal** | Bushwick | `not_found` | `None` | None |
| **Rio Hotel & Casino** | Las Vegas | `not_found` | `api` | None |
| **Salon on Kingston** | Crown Heights | `not_found` | `None` | None |
| **The Backyards** | Bedford-Stuyvesant | `KovZpZAIkJIA` | `None` | None |
| **The Dram Shop Bar** | Park Slope | `not_found` | `None` | None |
| **The Quays** | Astoria | `not_found` | `None` | None |
| **The ROGUE** | Park Slope | `KovZpZAatAlA` | `None` | None |
| **The Rockwell Place** | Downtown Brooklyn | `not_found` | `None` | [Events Link](https://www.therockwellplace.com) |
| **The VSPOT Restaurant** | Park Slope | `not_found` | `None` | [Website](https://vspot.nyc/) |
| **Two Boots Williamsburg** | Williamsburg | `not_found` | `None` | None |
| **Village Lantern** | Greenwich Village | `KovZpZAFlntA` | `None` | None |
| **Village Underground** | Greenwich Village | `not_found` | `api` | [Events Link](https://www.comedycellar.com/new-york-line-up/) |

### community centers with events (22 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Al Oerter Recreation Center** | Queens | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Carmine Carro Community Center** | Marine Park | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Chelsea Recreation Center** | Chelsea | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Conference House Museum** | Staten Island | `None` | `None` | [Events Link](https://conferencehouse.org/events/) |
| **Conference House Park** | Staten Island | `None` | `scrape` | [Events Link](https://conferencehouse.org/events/) |
| **Constance Baker Motley Recreation Center** | Manhattan | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Crotona Nature Center** | Bronx | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Crotona Park Community Center** | Crotona Park, Bronx | `None` | `None` | [Events Link](http://www.phippsny.org/events) |
| **Cunningham Park** | Queens | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Herbert Von King Cultural Arts Center** | Bedford-Stuyvesant | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Inwood Hill Park** | Inwood | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/parks/inwood-hill-park/events) |
| **John Malone Community Center** | McGuire Park, Brooklyn | `None` | `None` | None |
| **King Manor Museum** | Jamaica | `None` | `None` | [Events Link](https://www.kingmanor.org/Events) |
| **Kingsbridge Heights Community Center** | Kingsbridge Heights, Bronx | `None` | `None` | [Events Link](https://www.khcc-nyc.org/events) |
| **Kwame Ture Recreation Center** | Manhattan | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **McKinley Park** | Brooklyn | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Orchard Beach Nature Center** | Bronx | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Pelham Fritz Recreation Center** | Manhattan | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Red Hook Recreation Center** | Red Hook | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Stuyvesant Square** | Manhattan | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Sunset Park Recreation Center** | Sunset Park | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Supportive Children's Advocacy Center - New York** | Bronx | `None` | `None` | [Events Link](http://www.scanny.org/events) |

### nyc parks with free events (21 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Adam Clayton Powell Jr. Malls** | NY 10027 | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Andrew Haswell Green Park** | FDR Dr | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Asser Levy Playground** | E 25th St | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Brownsville Playground** | Sackman St | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Cherry Tree Park** | New York | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **East River Waterfront Esplanade** | FDR Dr | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Fidler-Wyckoff House Park** | Brooklyn | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Gertrude Ederle Recreation Center** | New York | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Hart Island** | Bronx | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **LaTourette Park &amp; Golf Course** | Staten Island | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Livonia Park** | Powell St | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Msgr. McGolrick Park** | Nassau Ave | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **North Shore Esplanade** | Unknown | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Playground Sixty Two LXII** | 108th St | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Powell's Cove Park** | Whitestone | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Rev. T. Wendell Foster Park and Recreation Center** | Bronx | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Roger Morris Park** | New York | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Rufus King Park** | Jamaica | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Seth Low Playground/ Bealin Square** | Bay Parkway | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **South Pacific Playground** | Howard Ave | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **St. John's Park** | Brooklyn | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |

### movie theaters repertory cinema (20 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Beekman Theatre** | Manhattan | `None` | `None` | None |
| **Brooklyn Public Library (Central Library, Dweck Center)** | Prospect Heights | `None` | `None` | None |
| **Cinema 1, 2 & 3 by Angelika** | Manhattan | `None` | `api` | [Website](https://www.angelikafilmcenter.com/nyc/cinema-1-2-3) |
| **City Cinemas Beekman Theatre** | Manhattan | `None` | `None` | None |
| **Film at Lincoln Center** | Upper West Side | `None` | `None` | [Events Link](https://www.filmlinc.org/events.html) |
| **Fine Arts Theatre** | Manhattan | `None` | `None` | None |
| **Kew Gardens Cinemas** | Kew Gardens | `None` | `None` | None |
| **Lincoln Plaza Cinemas** | Upper West Side | `None` | `None` | None |
| **MoMA Film** | Manhattan | `None` | `None` | [Events Link](https://www.moma.org/calendar/events/11201) |
| **Morbid Anatomy Museum** | Gowanus | `None` | `None` | None |
| **Museum of the Moving Image** | Astoria | `None` | `None` | [Events Link](https://movingimage.us/events.html) |
| **Producers Club Theaters** | Hell's Kitchen | `None` | `None` | [Events Link](https://producersclub.com/shows/) |
| **Rooftop Cinema Club Midtown** | Midtown | `None` | `None` | [Events Link](https://rooftopcinemaclub.com/new-york/midtown/) |
| **Spectacle Theater** | Williamsburg | `None` | `None` | [Events Link](https://www.spectacletheater.com/the-scott-and-gary-show/) |
| **Syndicated Bar Theater Kitchen** | Bushwick | `None` | `None` | [Events Link](https://syndicatedbk.com/) |
| **The Landmark at 57 West** | Manhattan | `None` | `api` | None |
| **Theater 80 at St Marks Place** | East Village | `None` | `None` | None |
| **Theनरी Theatre** | Midtown | `None` | `None` | None |
| **Ziegfeld Theatre (1969)** | Manhattan | `None` | `None` | None |
| **reRun Gastropub Theater** | Brooklyn | `None` | `None` | None |

### rock music venues indie (18 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **18th Ward Brewing** | East Williamsburg | `rZ7HnEZ17_3AN` | `None` | None |
| **Bar Laika** | Clinton Hill | `not_found` | `None` | None |
| **Berlin - Under A** | East Village | `not_found` | `None` | [Website](https://www.berlinundera.com/) |
| **Bohemian Grove** | Bushwick | `KovZpZAFEatA` | `None` | None |
| **East Williamsburg Econolodge** | East Williamsburg | `not_found` | `None` | None |
| **Flowers for all Occasions** | Bushwick | `not_found` | `None` | None |
| **FourFiveSix** | Williamsburg | `not_found` | `None` | None |
| **Hell Phone** | Bushwick | `not_found` | `None` | None |
| **Main Drag Music** | Williamsburg | `not_found` | `None` | [Events Link](https://www.maindragmusic.com/) |
| **Our Place** | Ridgewood | `not_found` | `None` | None |
| **Rockwood Music Hall - Stage 1** | Lower East Side | `not_found` | `None` | None |
| **The Glove** | Bushwick | `not_found` | `None` | None |
| **The Grove** | Bushwick | `rZ7HnEZ17_K-K` | `None` | None |
| **The Kingsland** | Greenpoint | `KovZpZA6dEkA` | `None` | None |
| **The Monarch** | East Williamsburg | `KovZpZAaettA` | `None` | None |
| **The Silent Barn** | Bushwick | `KovZpZAknI7A` | `None` | None |
| **Unit J** | Bushwick | `not_found` | `None` | None |
| **Windjammer Bar** | Ridgewood | `not_found` | `None` | None |

### photography galleries (18 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Bronx Documentary Center** | Melrose | `None` | `None` | [Events Link](https://bronxdoc.org/events/) |
| **Club Rhubarb** | Lower East Side | `None` | `None` | None |
| **Dempsey Theater in Harlem** | Harlem | `None` | `None` | None |
| **Dwyer Cultural Center** | Harlem | `None` | `None` | None |
| **Faison Firehouse Theater** | Harlem | `None` | `None` | None |
| **Harlem Heritage Tourism and Cultural Center** | Harlem | `None` | `None` | [Events Link](https://www.harlemheritage.com/wp-json/tribe/events/v1/) |
| **Harlem Repertory Theatre** | East Harlem | `None` | `None` | None |
| **International Center of Photography (ICP)** | Lower East Side | `None` | `None` | [Events Link](http://www.icp.org/events/winter-2026-exhibitions-tour-february-20) |
| **Julia De Burgos Performance & Arts Center** | East Harlem | `None` | `None` | None |
| **LeRoy Neiman Art Center** | Harlem | `None` | `None` | None |
| **Leslie-Lohman Museum of Art** | SoHo | `None` | `None` | [Events Link](https://leslielohman.org/exhibitions/pamela-sneed-and-carlos-martiel-sacred-and-profane) |
| **Museum of Art and Origins** | Washington Heights | `None` | `None` | None |
| **Museum of Contemporary Photography** | South Loop | `None` | `None` | None |
| **National Jazz Museum in Harlem** | Harlem | `None` | `None` | None |
| **Revolutionary Theatre Company** | Harlem | `None` | `None` | None |
| **Sugar Hill Children's Museum of Art & Storytelling** | Sugar Hill | `None` | `None` | [Events Link](https://www.sugarhillmuseum.org/programs-1) |
| **The Firehouse Theatre** | East Harlem | `None` | `None` | None |
| **The Museum of Modern Art (MoMA)** | Midtown | `None` | `None` | [Events Link](https://www.moma.org/calendar/exhibitions/history) |

### event spaces party venues (17 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **3 West Club** | Midtown | `None` | `None` | [Events Link](https://www.3westclub.com/events/) |
| **Atlantis Hall** | Richmond Hill | `None` | `None` | [Events Link](https://atlantishall.com/) |
| **Bronx House Community Center** | Pelham Parkway | `None` | `None` | [Events Link](https://www.bronxhouse.org/) |
| **Canary Club** | Lower East Side | `None` | `None` | [Events Link](https://canaryclubnyc.com/) |
| **Edgewater Hall** | Stapleton | `None` | `None` | None |
| **Greenbelt Recreation Center** | Greenbelt | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **JCC Cornerstone Community Center at Richmond Terrace** | New Brighton | `None` | `None` | None |
| **Loosie's Nightclub** | Lower East Side | `None` | `None` | [Website](https://loosiesnyc.com/) |
| **MJ Catering Hall** | Highbridge | `None` | `None` | None |
| **MME Banquet Hall** | Parkchester | `None` | `None` | None |
| **Marte Hall** | NYC | `None` | `None` | None |
| **Phipps Neighborhoods Community Center at Davidson Houses** | Morrisania | `None` | `None` | None |
| **Phipps Neighborhoods Community Center at Sotomayor Houses** | Soundview | `None` | `None` | None |
| **Phipps Neighborhoods Community Center at Soundview Houses** | Soundview | `None` | `None` | None |
| **RV Party Hall** | Soundview | `None` | `None` | None |
| **The Banquet Hall at Nansen Park** | Travis-Chelsea | `None` | `None` | None |
| **The Times Center** | Midtown Manhattan | `None` | `None` | [Website](https://www.thetimescenter.com/) |

### history museums NYC (14 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **9/11 Tribute Museum** | Lower Manhattan | `None` | `None` | None |
| **Federal Hall National Memorial Visitor** | NYC | `None` | `None` | [Events Link](https://www.nps.gov/planyourvisit/event-search.htm) |
| **General Grant National Memorial** | NYC | `None` | `None` | [Events Link](https://www.nps.gov/gegr/planyourvisit/calendar.htm) |
| **Ground Zero Museum Workshop** | NYC | `None` | `None` | [Events Link](https://www.groundzeromuseumworkshop.org/tickets.html) |
| **Hamilton Grange National Memorial** | Manhattan | `None` | `None` | [Events Link](https://www.nps.gov/hagr/planyourvisit/calendar.htm) |
| **Liberty Island Information Center - Statue of Liberty National Monument** | NYC | `None` | `None` | [Events Link](https://www.nps.gov/stli/planyourvisit/calendar.htm) |
| **Lower East Side Tenement Museum** | Lower East Side | `None` | `None` | [Events Link](https://www.tenement.org/events.html) |
| **Museum At Eldridge Street** | NYC | `None` | `None` | [Events Link](https://www.eldridgestreet.org/events) |
| **Museum of American Finance** | NYC | `None` | `None` | [Events Link](https://www.moaf.org/events/index) |
| **Museum of the American Gangster** | East Village | `None` | `None` | None |
| **New-york Historical Society** | NYC | `None` | `None` | [Events Link](https://www.nyhistory.org/programs) |
| **St. Patrick's Cathedral** | NYC | `None` | `None` | [Events Link](https://www.saintpatrickscathedral.org/upcoming-events) |
| **Statue of Liberty** | NYC | `None` | `None` | [Events Link](https://www.nps.gov/stli/planyourvisit/calendar.htm) |
| **Theodore Roosevelt Birthplace National Historic Site** | Flatiron District | `None` | `None` | [Events Link](https://www.nps.gov/thrb/planyourvisit/calendar.htm) |

### Shakespeare in the Park NYC (12 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Alley Pond Park** | Oakland Gardens | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Bronx Sunshine Community Garden** | West Farms | `None` | `None` | None |
| **Brownsville Recreation Center** | Brownsville | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **East River Park Amphitheater** | Lower East Side | `None` | `None` | None |
| **Herbert Von King Park Amphitheater** | Bedford-Stuyvesant | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Parking Lot behind The Clemente Soto Velez Cultural and Educational Center** | Lower East Side | `None` | `None` | [Website](https://clemente.org/) |
| **Richard Rodgers Amphitheater in Marcus Garvey Park** | Harlem | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **Roberto Clemente Community Garden** | Highbridge | `None` | `None` | None |
| **Soldiers' and Sailors' Monument (Hudson Classical Theater Company)** | Upper West Side | `None` | `None` | None |
| **Staten Island Borough Hall** | St. George | `None` | `None` | [Events Link](https://www.statenislandusa.com/community-events.html) |
| **Target East Harlem Community Garden** | East Harlem | `None` | `None` | [Events Link](https://www.nyrp.org/en/gardens/target-east-harlem-community-garden/events.html) |
| **William Rainey Garden** | Longwood | `None` | `None` | None |

### improv comedy theaters (12 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Bushwick Comedy Club** | Bushwick | `None` | `None` | [Events Link](https://www.bushwickcomedyclub.com/) |
| **Comedy Village** | Hell's Kitchen | `None` | `None` | None |
| **Dangerfield's Comedy Club** | Upper East Side | `None` | `None` | None |
| **National Comedy Theatre** | Midtown | `None` | `None` | None |
| **Old Man Hustle BKLYN Comedy Club** | Williamsburg | `None` | `None` | None |
| **Old Man Hustle BKLYN Comedy Club & Bar** | Williamsburg | `None` | `None` | None |
| **Rodney's Comedy Club** | Upper East Side | `None` | `None` | [Website](https://rodneyscomedy.com/) |
| **Sesh Comedy** | Lower East Side | `None` | `None` | None |
| **Stones Comedy Club** | Midtown East | `None` | `None` | None |
| **The Annoyance Theatre NY** | Williamsburg | `None` | `None` | None |
| **The Grisly Pear Comedy Club** | Greenwich Village | `None` | `None` | [Website](https://www.grislypearcomedy.com/) |
| **The Nest Theatre** | Prospect Lefferts Gardens | `None` | `None` | None |

### immersive theater venues (12 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Arte Museum** | Chelsea | `None` | `None` | [Events Link](https://artemuseum.com/) |
| **Banksy Museum** | SoHo | `None` | `None` | None |
| **Jalopy Theatre and School of Music** | Columbia Street Waterfront District | `None` | `None` | [Events Link](https://www.jalopytheatre.org/) |
| **Jamaica Performing Arts Center (JPAC)** | Jamaica | `None` | `None` | None |
| **Jekyll and Hyde Club** | West Village | `None` | `None` | None |
| **Masquerade: The Phantom of the Opera** | East Village | `None` | `None` | None |
| **Perelman Performing Arts Center (PAC NYC)** | Financial District | `None` | `None` | [Events Link](https://pacnyc.org/calendar/) |
| **The Glow Cultural Center** | Flushing | `None` | `None` | [Events Link](https://glownyc.org/calendar/month/) |
| **The Owl Music Parlor** | Prospect Lefferts Gardens | `None` | `None` | [Events Link](https://theowl.nyc/calendar/) |
| **The Ruby Theatre** | Midtown | `None` | `None` | None |
| **The Secret Theatre / Secret Arts** | Long Island City | `None` | `None` | [Events Link](https://secrettheatre.com/events) |
| **Thespis Theater** | Long Island City | `None` | `None` | None |

### ice skating rinks (11 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Brewster Ice Arena** | Brewster | `None` | `None` | [Events Link](https://www.brewstericearena.com/development-programs) |
| **Clark Gillies Arena** | Dix Hills | `None` | `None` | [Events Link](https://www.huntingtonny.gov/ice-rink) |
| **Clary Anderson Arena** | Montclair | `None` | `None` | [Website](https://claryandersonarena.com/) |
| **Floyd Hall Arena** | Little Falls | `None` | `None` | None |
| **Long Beach Arena** | Long Beach | `None` | `None` | None |
| **Mennen Sports Arena** | Morristown | `None` | `None` | None |
| **Murray's Skating Center** | Yonkers | `None` | `None` | None |
| **Palisades Center Ice Rink** | West Nyack | `None` | `None` | None |
| **Port Washington Skating Center** | Port Washington | `None` | `None` | [Events Link](https://pwskating.com/) |
| **Town of Oyster Bay Ice Skating Center** | Bethpage | `None` | `None` | [Events Link](https://oysterbaytown.com/departments/parks/ice-skating/) |
| **Winter Garden Ice Arena** | Ridgefield | `None` | `None` | None |

### alternative performance spaces (11 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Chain Theatre - Black Box** | Midtown | `None` | `None` | [Events Link](https://www.chaintheatre.org) |
| **Club 57 (Historical)** | East Village | `None` | `None` | None |
| **Mudd Club (Historical)** | TriBeCa | `None` | `None` | None |
| **Ontological-Hysteric Theater** | East Village | `None` | `None` | None |
| **Pyramid Club (Historical)** | East Village | `None` | `None` | None |
| **Rockwood Music Hall - Stage 3** | Lower East Side | `None` | `None` | None |
| **The Living Theatre** | Lower East Side | `None` | `None` | [Website](https://www.livingtheatre.org/) |
| **The Peoples Improv Theater - Simple Space** | Flatiron District | `None` | `None` | None |
| **The Players Theatre - Steve & Marie Sgouros Theatre** | Greenwich Village | `None` | `None` | [Website](https://playerstheatre.com/) |
| **The Players Theatre - The Steve & Marie Sgouros Theatre** | Greenwich Village | `None` | `None` | [Events Link](https://playerstheatre.com/) |
| **The Producer's Club Theaters & Bar** | Hell's Kitchen | `None` | `None` | [Events Link](https://producersclub.com/shows/) |

### brooklyn music venues nightclubs (9 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Bad Therapy** | Kensington | `not_found` | `None` | None |
| **H I D D E N** | Bed-Stuy | `KovZpafPke` | `None` | None |
| **Lowlands** | Brooklyn | `not_found` | `None` | None |
| **Moxy Williamsburg** | Williamsburg | `not_found` | `None` | [Events Link](https://www.marriott.com/en-us/hotels/nycmw-moxy-brooklyn-williamsburg/overview/events) |
| **Sunny’s Bar** | Red Hook | `not_found` | `None` | [Events Link](https://www.sunnysredhook.com/calendar2/) |
| **The Brooklyn Monarch** | Williamsburg | `KovZpZAaettA` | `None` | [Website](https://www.thebrooklynmonarch.com/) |
| **Union Pool** | Brooklyn | `Z7r9jZadjd` | `None` | [Website](https://www.union-pool.com/) |
| **Unveiled** | Williamsburg | `Z7r9jZaAHf` | `None` | [Website](https://unveiledbrooklyn.com/) |
| **Velvet Brooklyn** | Williamsburg | `not_found` | `None` | [Events Link](https://velvetbrooklyn.com/) |

### museums (9 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Brooklyn Art Library** | Williamsburg | `None` | `None` | None |
| **Building 92 at the Brooklyn Navy Yard** | Brooklyn Navy Yard | `None` | `None` | None |
| **Cultural Lab LIC** | Long Island City | `None` | `None` | [Website](https://www.culturallab.org/) |
| **Edward Mooney House** | Chinatown | `None` | `None` | None |
| **Kreuzer-Pelton House** | West New Brighton | `None` | `None` | None |
| **Lehman College Art Gallery** | Bedford Park | `None` | `None` | None |
| **Museum of Ice Cream** | SoHo | `None` | `None` | [Events Link](https://www.museumoficecream.com/events/) |
| **Museum of Sex** | Nomad | `None` | `None` | [Events Link](https://www.museumofsex.com/events/) |
| **National Museum of the American Indian - New York** | Financial District | `None` | `None` | None |

### warehouse music venues brooklyn (8 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Box of Moonlight** | Bed-Stuy | `not_found` | `None` | None |
| **HOOK Studios** | Red Hook | `not_found` | `None` | [Events Link](https://hook-studios.com/) |
| **ML Downtown Brooklyn** | Gowanus | `not_found` | `None` | None |
| **Sander Studios** | Clinton Hill | `not_found` | `None` | [Website](https://www.sanderstudios.com/) |
| **Shapira's Assemble Warehouse** | Greenpoint | `not_found` | `None` | None |
| **Shapira's RAW Warehouse** | Williamsburg | `not_found` | `None` | None |
| **The 1896 Studios & Stages** | Bushwick | `KovZ917AxWi` | `None` | [Website](https://the1896.com/) |
| **Wandering Barman** | Williamsburg | `not_found` | `None` | [Events Link](https://wanderingbarman.com/) |

### bars with live events comedy (8 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Backroom Comedy** | East Village | `None` | `None` | None |
| **Barbershop Comedy Show at The Original Barbershop** | Lower East Side | `None` | `None` | None |
| **Bobo's Comedy Club** | New Dorp | `None` | `None` | None |
| **Comedy UO** | Various neighborhoods in Manhattan | `None` | `None` | None |
| **Flop House Comedy Club** | Williamsburg | `None` | `None` | [Website](https://www.flophousecomedy.com/) |
| **High Line Comedy Club** | Meatpacking District | `None` | `None` | None |
| **Sheba's Speakeasy Comedy Club** | Midtown | `None` | `None` | None |
| **The Top Secret Comedy Club** | East Village | `None` | `None` | None |

### park (8 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Central Park** | Manhattan | `None` | `scrape` | [Events Link](https://www.centralparknyc.org/) |
| **Crotona Park** | Bronx | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Flushing Meadows Corona Park** | Flushing | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Herbert Von King Park** | Bedford-Stuyvesant | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Hunts Point Recreation Center** | Bronx | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |
| **McCarren Park** | Williamsburg | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Shakespeare Garden** | Upper West Side | `None` | `None` | [Events Link](https://www.centralparknyc.org/activities/events) |
| **St. James Recreation Center** | Bronx | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |

### fitness class studios (7 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **AK Boxing Club** | Financial District | `None` | `None` | [Events Link](https://akboxing.com/) |
| **Bottega Bellantuono Ballroom Dance Studio** | Manhattan | `None` | `None` | [Events Link](https://bottegabellantuono.com/) |
| **Club 300 at Theradynamics** | Tribeca | `None` | `None` | [Events Link](https://www.club300nyc.com/) |
| **Mile High Run Club** | NoHo | `None` | `None` | [Events Link](https://www.milehighrunclub.com/) |
| **NYC Volleyball Academy** | Financial District | `None` | `None` | [Website](https://www.nycvolleyballacademy.com/) |
| **Peloton Studio NYC** | Hudson Yards | `None` | `api` | [Website](https://studio.onepeloton.com/new-york/schedule) |
| **Trinity Boxing Club - New York** | Lower Manhattan | `None` | `None` | [Events Link](https://trinityboxing.com/) |

### libraries with public events (7 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **American Kennel Club Library & Archives** | Midtown | `None` | `None` | None |
| **Brooklyn Historical Society (now Center for Brooklyn History)** | Brooklyn Heights | `None` | `None` | [Events Link](https://www.bklynlibrary.org/cbh/education/events) |
| **Carroll Gardens Pop-Up Library** | Carroll Gardens | `None` | `None` | None |
| **Conjuring Arts Research Center** | Midtown | `None` | `None` | [Events Link](https://conjuringarts.org/category/exhibitions/) |
| **International Center of Photography Library** | Lower East Side | `None` | `None` | None |
| **The Explorers Club** | Upper East Side | `None` | `None` | [Events Link](https://www.explorers.org/wp-json/tribe/events/v1/) |
| **Yeshiva University Museum** | Chelsea | `None` | `None` | [Events Link](https://www.yumuseum.org/exhibitions/current) |

### unique event venues (7 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Edison Ballroom** | Midtown | `None` | `None` | [Website](https://edisonballroom.com/) |
| **Jazz at Lincoln Center's Frederick P. Rose Hall** | Upper West Side | `None` | `None` | [Events Link](https://jazz.org/watch-listen-discover/program-archive/) |
| **Metropolitan Pavilion** | Chelsea | `None` | `None` | None |
| **The Prince George Ballroom** | NoMad | `None` | `None` | [Events Link](https://www.princegeorgeballroom.org/) |
| **The Water Club** | Kips Bay | `None` | `None` | None |
| **Vista Penthouse Ballroom & Sky Lounge** | Long Island City | `None` | `None` | [Website](https://vistanyc.com/) |
| **Ziegfeld Ballroom** | Midtown | `None` | `None` | [Website](https://ziegfeldballroom.com/) |

### best nightclubs (6 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **40/40 Club** | Flatiron District | `None` | `None` | None |
| **Amadeus Nightclub** | Elmhurst | `None` | `None` | None |
| **Doha Nightclub** | Long Island City | `None` | `None` | None |
| **Mission Nightclub** | Chelsea | `None` | `None` | None |
| **Paradise Club** | Times Square | `None` | `None` | [Events Link](https://www.paradiseclubnyc.com/events.html) |
| **The Carnegie Club** | Midtown | `None` | `None` | None |

### art galleries (6 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Grey Art Museum (NY University)** | NoHo | `None` | `None` | None |
| **Marvin Gardens** | Ridgewood | `None` | `None` | None |
| **The Brant Foundation Art Study Center** | East Village | `None` | `None` | [Events Link](https://www.brantfoundation.org/events/) |
| **The Metropolitan Museum of Art** | NYC | `None` | `None` | [Events Link](https://www.metmuseum.org/exhibitions) |
| **The Museum of Modern Art** | NYC | `None` | `None` | [Events Link](https://www.moma.org/calendar/exhibitions/history) |
| **Woodward Gallery** | Lower East Side | `None` | `api` | [Events Link](https://woodwardgallery.net/richard-hambleton-momentum-selected-paintings-exhibition/) |

### parades and parade routes (6 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Bronx Halloween Parade** | Southern Boulevard | `None` | `None` | None |
| **Elmhurst Memorial Hall Wreath-Laying** | Elmhurst | `None` | `None` | None |
| **JASA Rockaway Park Older Adult Center Veterans Day Celebration** | Rockaway Park | `None` | `None` | [Events Link](https://www.jasa.org/events/art-show-2026) |
| **NYC Department of Veterans' Services Resource Center at Bronx Borough Hall** | Concourse | `None` | `None` | [Events Link](https://www.nyc.gov/main/events/) |
| **Tompkins Square Halloween Dog Parade** | East Village | `None` | `None` | [Website](https://www.tompkinssquaredogrun.com/) |
| **Triboro Center Veterans Day Ceremony** | Morrisania | `None` | `None` | [Events Link](https://triborocenter.net/events) |

### best comedy shows (5 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Basement Comedy Club** | Chinatown | `None` | `None` | None |
| **Caroline's Comedy Club** | Times Square | `None` | `None` | None |
| **Club Cummings** | East Village | `None` | `None` | None |
| **Old Man Hustle Comedy Bar** | Lower East Side | `None` | `None` | None |
| **The Duplex Cabaret Theatre** | Greenwich Village | `None` | `None` | [Events Link](https://www.theduplex.com/) |

### trivia night bars (5 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Clinton Hall Bronx** | Belmont | `None` | `None` | None |
| **Gowanus Gardens** | Gowanus | `None` | `None` | None |
| **Hamilton Hall** | Harlem | `None` | `None` | None |
| **Radegast Hall & Biergarten** | Williamsburg | `None` | `None` | [Events Link](https://www.radegasthall.com/events) |
| **Valhalla Bar** | Hell's Kitchen | `None` | `None` | [Events Link](https://www.valhallabarnyc.com/) |

### warehouse party venues brooklyn (4 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Black Box Theater with Sunny Courtyard and Lush Garden** | Greenpoint | `None` | `None` | [Events Link](https://themusebrooklyn.com/) |
| **Brooklyn Expo Center** | Greenpoint | `None` | `None` | None |
| **Gowanus Ballroom** | Gowanus | `None` | `None` | None |
| **Kings Hall** | East Williamsburg | `None` | `None` | None |

### public plazas with programming (4 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Jamaica Center Plaza** | Jamaica | `None` | `None` | None |
| **Sunnyside Gardens Plaza** | Sunnyside | `None` | `None` | None |
| **Verdi Square** | Upper West Side | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Water and Whitehall Plaza** | Financial District | `None` | `None` | None |

### magic show venues (3 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Asi Wind's Inner Circle at The Judson Theatre** | Greenwich Village | `None` | `None` | None |
| **The Parlour of Deceptions at Salmagundi Club** | Greenwich Village | `None` | `None` | None |
| **The Quantum Eye at Guild Hall** | NoMad | `None` | `None` | None |

### music venue (3 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Knockdown Center** | Maspeth | `KovZpZAEteAA` | `None` | [Website](https://knockdown.center/) |
| **Market Hotel** | Bushwick | `KovZpZA6AInA` | `None` | [Website](https://www.markethotel.org/) |
| **The Brooklyn Mirage & Avant Gardner** | East Williamsburg | `not_found` | `None` | [Events Link](https://www.avant-gardner.com/events) |

### art galleries brooklyn (3 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Dumbo Arts Center (DAC)** | DUMBO | `None` | `None` | None |
| **New Earth Museum** | Red Hook | `None` | `None` | None |
| **Orwell's Garden** | Williamsburg | `None` | `None` | None |

### outdoor venue (3 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Elizabeth Street Garden** | Nolita | `None` | `None` | [Events Link](https://www.elizabethstreetgarden.com/) |
| **Public Square & Gardens at Hudson Yards** | Hudson Yards | `None` | `None` | None |
| **Rockaway Beach and Boardwalk** | Rockaway | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |

### best things to do at night (3 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Hamilton at Richard Rodgers Theatre** | Theater District | `None` | `None` | None |
| **LAVO Nightclub** | Midtown East | `None` | `None` | [Events Link](https://taogroup.com/venues/lavo-nightclub-new-york/events.html) |
| **Wicked at Gershwin Theatre** | Theater District | `None` | `None` | None |

### science museums NYC (3 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Hayden Planetarium** | Upper West Side | `None` | `None` | [Events Link](https://www.amnh.org/exhibitions/hayden-planetarium/events.html) |
| **Museum of the Peaceful Arts** | Manhattan | `None` | `None` | None |
| **New York Museum of Science and Industry** | Midtown | `None` | `None` | None |

### burlesque drag show venues (3 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Laurie Beechman Theatre** | Midtown West | `None` | `None` | [Events Link](https://www.wbcnyc.com/) |
| **Rosewood Theater** | Hell's Kitchen | `None` | `None` | None |
| **St. Mazie Bar & Supper Club** | Williamsburg | `None` | `None` | [Events Link](https://www.stmazie.com/events) |

### art galleries chelsea (2 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Amsterdam Whitney Gallery** | Chelsea | `None` | `None` | [Events Link](https://www.amsterdamwhitneygallery.com/copy-of-current-exhibition) |
| **Firehouse: A Center for the Arts and Civil Rights** | Bedford-Stuyvesant | `None` | `None` | None |

### theater (2 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Billie Holiday Theatre** | Bedford-Stuyvesant | `None` | `None` | [Website](https://thebillieholiday.org/) |
| **Triple Promise Academy for the Performing Arts** | Bay Ridge | `None` | `None` | [Events Link](https://triplepromise.com/shows/) |

### rooftop bars with events (2 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Kimoto Rooftop Restaurant & Garden Lounge** | Downtown Brooklyn | `None` | `None` | None |
| **The Cantor Roof Garden Bar** | Upper East Side | `None` | `None` | None |

### parks with events concerts NYC (2 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Forest Park** | Woodhaven | `None` | `scrape` | [Events Link](https://www.nycgovparks.org/events) |
| **Williamsbridge Oval Recreation Center** | Bronx | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |

### bar with events (2 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Jean’s** | NoHo | `None` | `api` | [Website](https://www.jeans.nyc/) |
| **Westlight** | Williamsburg | `None` | `api` | [Website](https://www.westlightnyc.com/) |

### circus aerial performance venues (2 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Joe's Pub at The Public Theater** | NoHo | `None` | `None` | [Events Link](https://publictheater.org/visit/joes-pub/events.html) |
| **Manhattan Movement Arts Center** | Upper West Side | `None` | `None` | [Events Link](http://www.myfirstexhusband.com/) |

### broadway theaters (2 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Niblo's Garden** | SoHo | `not_found` | `None` | None |
| **Park Theatre** | Financial District | `KovZ917AYLO` | `None` | None |

### outdoor event spaces gardens NYC (1 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **620 Loft & Garden** | Midtown Manhattan | `None` | `None` | [Events Link](https://www.rockefellercenter.com/private-events/620-loft-and-garden/events.html) |

### podcast recording venues (1 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Blitz Club Studios** | Ridgewood | `None` | `None` | [Website](https://blitz.nyc/) |

### bookstores with author events (1 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Bluestockings Cooperative Bookstore** | Lower East Side | `None` | `None` | [Website](https://bluestockings.com/) |

### dance venue (1 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Musica** | Hell’s Kitchen | `None` | `None` | [Events Link](https://www.musicanewyork.com/events) |

### summer concerts parks NYC (1 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Northwell Health at Jones Beach Theater** | Long Island | `None` | `None` | None |

### variety venue (1 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **NYC Parks Recreation Centers** | NYC | `None` | `None` | [Events Link](https://www.nycgovparks.org/events) |

### outdoor concert venues amphitheaters stadiums (1 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **PNC Bank Arts Center** | Holmdel | `not_found` | `None` | None |

### documentary film venues (1 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **SVA Theatre** | Chelsea | `None` | `None` | [Events Link](https://svatheatre.com/events/bfa-animation-faculty-show-tell-between-teaching-making/) |

### cultural center (1 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Vera List Center for Art and Politics** | NYC | `None` | `None` | [Events Link](https://veralistcenter.org/events/robert-rauschenberg-the-news-inaugural-panel) |

### event space (1 venues)

| Venue Name | Neighborhood | Ticketmaster ID | Preferred Source | Website / Events URL |
|---|---|---|---|---|
| **Wythe Hotel - Main Hall** | Williamsburg | `None` | `None` | None |

## Other Venues with 0 Events
There are 2383 other venues in the database (such as small neighborhood parks, local community gardens, or minor plazas) that currently have 0 events. These are likely low-priority or seasonal spaces.
