"""polaris_rag.retrieval.document_loader

Document loading utilities for multiple data sources.

This module provides helpers for acquiring raw documents from:
- Websites (HTML pages)
- Jira (support tickets)

The primary outputs are :class:`polaris_rag.common.schemas.Document` objects
for web content, and raw Jira issue dictionaries for tickets (which are
typically preprocessed into :class:`~polaris_rag.common.schemas.Document`
instances elsewhere).

Functions
---------
is_internal_link
    Check whether a hyperlink is internal relative to a base URL.
remove_link_fragment
    Remove the fragment component (``#...``) from a URL.
get_internal_links
    Extract internal links from an HTML page.
load_website_docs
    Fetch HTML pages and return them as :class:`~polaris_rag.common.schemas.Document` objects.
create_jira_jql_query
    Build a JQL query string for resolved Jira issues over a date window.
load_support_tickets
    Retrieve resolved Jira issues over a date window using configured credentials.
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urlsplit, urlunsplit
from atlassian import Jira
from datetime import datetime

from polaris_rag.common import Document
from polaris_rag.config import GlobalConfig

DEFAULT_CHARSET = "UTF-8"

def is_internal_link(link: str,
                base_url: str,
                allow_subdomains: bool = True) -> bool:
    """Check whether a hyperlink is internal relative to a base URL.

    A link is considered internal if it resolves to the same host as ``base_url``.
    When ``allow_subdomains`` is ``True``, subdomains of the base host are also
    treated as internal.

    Non-HTTP(S) schemes (e.g., ``mailto:``, ``tel:``, ``javascript:``) are treated
    as external.

    Parameters
    ----------
    link : str
        The hyperlink URL to check. This may be relative or absolute.
    base_url : str
        Base page URL used to resolve relative links and define the host boundary.
    allow_subdomains : bool, optional
        Whether subdomains of the base host should be treated as internal.
        Defaults to ``True``.

    Returns
    -------
    bool
        ``True`` if the resolved link is internal, otherwise ``False``.
    """
    resolved = urlparse(urljoin(base_url, link))
    base     = urlparse(base_url)

    if resolved.scheme not in ("http", "https"):
        return False

    if not resolved.hostname:
        return True

    if resolved.hostname == base.hostname:
        return True

    if allow_subdomains and base.hostname and resolved.hostname.endswith("." + base.hostname):
        return True

    return False

def remove_link_fragment(link: str) -> str:
    """Remove the fragment component from a URL.

    Parameters
    ----------
    link : str
        URL that may contain a fragment (``#...``).

    Returns
    -------
    str
        URL with the fragment removed.
    """
    parts = urlsplit(link)

    return urlunsplit(parts._replace(fragment=""))


def get_internal_links(webpage_url: str) -> list[str]:
    """Extract internal links from a webpage.

    This function downloads ``webpage_url``, parses all ``<a href="...">`` links,
    filters them to those considered internal by :func:`is_internal_link`, removes
    fragments via :func:`remove_link_fragment`, and returns a de-duplicated list
    including ``webpage_url`` itself.

    Parameters
    ----------
    webpage_url : str
        URL of the page to fetch and parse.

    Returns
    -------
    list[str]
        De-duplicated list of internal URLs. Returns an empty list if the page
        cannot be fetched.
    """
    try:
        response = requests.get(webpage_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Could not access {webpage_url}: {e}")
        return []

    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all('a')
    documentation_urls = [webpage_url]

    for link in links:
        url = link.get('href', None)

        if url and is_internal_link(link=url, base_url=webpage_url):
            url = remove_link_fragment(url)

            documentation_urls.append(urljoin(webpage_url, url))

    return list(set(documentation_urls))

def load_website_docs(urls: list[str]) -> list[Document]:
    """Fetch HTML pages and return them as :class:`~polaris_rag.common.schemas.Document` objects.

    For each URL, the page is fetched via HTTP GET and the response body is
    decoded into text. Each returned :class:`~polaris_rag.common.schemas.Document`
    has ``document_type="html"`` and a ``metadata`` entry containing the source URL.

    Parameters
    ----------
    urls : list[str]
        List of webpage URLs to retrieve.

    Returns
    -------
    list[Document]
        List of documents corresponding to successfully fetched pages. URLs that
        cannot be retrieved are skipped.
    """
    documents = []

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Skipping {url}: {e}")
            continue

        html_content = response.content

        soup = BeautifulSoup(html_content, 'html.parser')

        charset = DEFAULT_CHARSET
        tag = soup.find('meta', charset=True)
        if tag and tag.get('charset'):
            charset = tag['charset']

        try:
            html_text = html_content.decode(charset, errors="replace")
        except LookupError:
            html_text = html_content.decode(DEFAULT_CHARSET, errors="replace")

        documents.append(
            Document(
                document_type = 'html',
                text = html_text,
                metadata = {
                    "source": url,
                }
            )
        )

    return documents

def create_jira_jql_query(
        start_date: str, 
        end_date: str, 
        keywords: list[str] | None = None,
        exclude_keys: list[str] | None = None,
        keys: list[str] | None = None,
        project_key: str = "HPCSSUP"
    ) -> str:
    """Build a JQL string for resolved issues within a created-date window.

    The generated query always filters by the given project and `status = Resolved`,
    applies an inclusive lower bound and an exclusive upper bound on the
    `created` field when `start_date` and/or `end_date` are provided, and orders
    results by most recent creation time.

    Parameters
    ----------
    start_date : str
        Inclusive lower bound for the `created` field in ISO date format
        ``YYYY-MM-DD``. If empty or ``None``, no lower bound is applied.
    end_date : str
        Exclusive upper bound for the `created` field in ISO date format
        ``YYYY-MM-DD``. If empty or ``None``, no upper bound is applied.
    keywords : list[str] | None, optional
        Optional list of keywords to filter issue summaries. If provided,
        only issues whose summary contains at least one keyword will be included.
        Defaults to ``None`` (no keyword filtering).
    exclude_keys : list[str] | None, optional
        Optional list of issue keys to exclude from results. If provided, any issue
        whose key matches one in this list will be filtered out. Defaults to ``None`` (no exclusions).
    keys : list[str] | None, optional
        Optional list of issue keys to include in results. If provided, only issues
        whose key matches one in this list will be included. Defaults to ``None`` (no key filtering).
    project_key : str, optional
        Jira project key to filter on. Defaults to ``"HPCSSUP"``.

    Returns
    -------
    str
        A JQL string of the form:
        ``project = '<project_key>' AND status = Resolved [AND created >= 'YYYY-MM-DD'] [AND created < 'YYYY-MM-DD'] ORDER BY created DESC``.

    Raises
    ------
    ValueError
        If ``start_date`` or ``end_date`` are provided but not in the
        ``YYYY-MM-DD`` format.

    Examples
    --------
    >>> create_jira_jql_query("2025-01-01", "2025-01-31", project_key="SUP")
    "project = 'SUP' AND status = Resolved AND created >= '2025-01-01' AND created < '2025-01-31' ORDER BY created DESC"
    """
    
    jql_query = (
        f"project = '{project_key}' "
        f"AND status = Resolved "
    )
    if start_date:
        try:
            parsed_date = None

            try:
                parsed_date = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                pass

            if parsed_date is None:
                raise ValueError(f"Invalid date format: {start_date}")
            
            formatted_date = parsed_date.strftime("%Y-%m-%d")
            jql_query += f"AND created >= '{formatted_date}' "
        except ValueError as e:
            raise ValueError(f"Invalid start_date provided: {start_date}") from e
        
    
    if end_date:
        try:
            parsed_date = None

            try:
                parsed_date = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                pass

            if parsed_date is None:
                raise ValueError(f"Invalid date format: {end_date}")
            
            formatted_date = parsed_date.strftime("%Y-%m-%d")
            jql_query += f"AND created < '{formatted_date}' "
        except ValueError as e:
            raise ValueError(f"Invalid end_date provided: {end_date}") from e
        
    keyword_clause = ""

    if keywords: 
        keyword_clause = " AND (" + " OR ".join([f'summary ~ "{keyword}"' for keyword in keywords]) + ")"
        jql_query += keyword_clause

    if exclude_keys:
        exclude_clause = " AND " + "key NOT IN (" + ", ".join([f'"{key}"' for key in exclude_keys]) + ")"
        jql_query += exclude_clause

    if keys:
        key_clause = " AND " + "key IN (" + ", ".join([f'"{key}"' for key in keys]) + ")"
        jql_query += key_clause
        
    jql_query += f"ORDER BY created DESC"

    return jql_query

def load_support_tickets(
        cfg: GlobalConfig = None,
        username: str = None,
        password: str = None,
        start_date: str = None, 
        end_date: str = None, 
        keywords: list[str] | None = None,
        exclude_keys: list[str] | None = None,
        keys: list[str] | None = None,
        project_key: str = "HPCSSUP",
        limit: int | None = 100,
    ) -> list[dict]:
    """Retrieve resolved Jira support tickets within a date range.

    This function connects to the Jira Cloud instance and retrieves issues from
    the specified project that are marked as *Resolved*. Authentication credentials
    are read from ``cfg`` via :attr:`polaris_rag.config.global_config.GlobalConfig.jira_api_credentials`.
    Results are fetched via :meth:`atlassian.Jira.enhanced_jql` with pagination.

    Parameters
    ----------
    start_date : str
        Inclusive lower bound for the issue creation date in ISO format
        (``YYYY-MM-DD``). If empty or ``None``, no lower bound is applied.
    end_date : str
        Exclusive upper bound for the issue creation date in ISO format
        (``YYYY-MM-DD``). If empty or ``None``, no upper bound is applied.
    cfg : GlobalConfig
        Loaded configuration containing Jira API credentials.
    keywords : list[str] | None, optional
        Optional list of keywords to filter issue summaries. If provided, 
        only issues whose summary contains at least one keyword will be 
        returned. Defaults to ``None`` (no keyword filtering).
    exclude_keys : list[str] | None, optional
        Optional list of issue keys to exclude from results. If provided, any issue
        whose key matches one in this list will be filtered out. Defaults to ``None`` (no exclusions).
    keys : list[str] | None, optional
        Optional list of issue keys to include in results. If provided, only issues
        whose key matches one in this list will be included. Defaults to ``None`` (no key filtering).
    project_key : str, optional
        Jira project key to query. Defaults to ``"HPCSSUP"``.
    limit : int or None, optional
        Maximum number of issues to retrieve. If ``None`` (default), all matching
        issues are returned.

    Returns
    -------
    list[dict]
        List of Jira issue dictionaries. Each issue contains standard Jira fields
        such as ``id``, ``key``, and ``fields``.

    Raises
    ------
    ValueError
        If ``start_date`` or ``end_date`` are provided but not valid ``YYYY-MM-DD`` strings.
    KeyError
        If Jira credentials are missing required keys (e.g., ``username`` or ``password``).
    """
    if cfg:
        config = cfg
        jira_api_config = config.jira_api_credentials

        username = jira_api_config['username']
        password = jira_api_config['password']
    if not username or not password:
        raise KeyError("Jira credentials are not properly configured.")

    jira = Jira(
        url='https://ucam-rcs.atlassian.net',
        username=username,
        password=password,
        cloud=True,
        api_version='3',
    )

    jql_query = create_jira_jql_query(
        start_date=start_date,
        end_date=end_date,
        keywords=keywords,
        exclude_keys=exclude_keys,
        keys=keys,
        project_key=project_key,
    )

    next_page_token = None
    issues = []

    while True:
        if limit:
            if len(issues) >= limit:
                break

        result = jira.enhanced_jql(
            jql_query,
            nextPageToken=next_page_token,
            expand=None,
        ) 

        temp_issues = result.get('issues', [])
        issues.extend(temp_issues)

        next_page_token = result.get('nextPageToken')

        if not next_page_token:
            break
    
    if limit is not None:
        issues = issues[:limit]

    return issues
