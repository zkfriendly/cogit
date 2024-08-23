from github import Github
import os
from github.GithubException import GithubException
import chardet
import json
import time

def fetch_github_data(repo_name, use_cache=True, cache_expiry=86400):  # 86400 seconds = 24 hours
    cache_file = f"{repo_name.replace('/', '_')}_cache.json"
    
    if use_cache and os.path.exists(cache_file):
        # Check if cache is still valid
        if time.time() - os.path.getmtime(cache_file) < cache_expiry:
            print(f"Using cached data for {repo_name}")
            with open(cache_file, 'r') as f:
                return json.load(f)
    
    print(f"Fetching data for {repo_name}")
    github_token = ""
    g = Github(github_token)
    documents = []
    
    try:
        repo = g.get_repo(repo_name)
        print(f"Successfully accessed repository: {repo_name}")
        
        # Load existing cache if it exists
        if os.path.exists(cache_file):
            print(f"Loading existing cache from {cache_file}")
            with open(cache_file, 'r') as f:
                documents = json.load(f)
            print(f"Loaded {len(documents)} documents from cache")
        
        # Fetch source code
        print("Fetching source code...")
        contents = repo.get_contents("")
        file_count = 0
        dir_count = 0
        while contents:
            file_content = contents.pop(0)
            try:
                if file_content.type == "dir":
                    print(f"Entering directory: {file_content.path}")
                    contents.extend(repo.get_contents(file_content.path))
                    dir_count += 1
                else:
                    try:
                        print(f"Processing file: {file_content.path}")
                        content = file_content.decoded_content
                        encoding = chardet.detect(content)['encoding']
                        decoded_content = content.decode(encoding or 'utf-8', errors='replace')
                        documents.append(f"File: {file_content.path}\n{decoded_content}")
                        file_count += 1
                        # Incrementally update cache
                        with open(cache_file, 'w') as f:
                            json.dump(documents, f)
                        print(f"Added file {file_content.path} to documents and updated cache")
                    except Exception as e:
                        print(f"Error decoding file {file_content.path}: {str(e)}")
            except GithubException as e:
                print(f"Error accessing content {file_content.path}: {str(e)}")
        print(f"Finished processing source code. Processed {file_count} files and {dir_count} directories.")
        
        # Fetch issues
        print("Fetching issues...")
        try:
            issue_count = 0
            for issue in repo.get_issues(state="all"):
                print(f"Processing issue #{issue.number}: {issue.title}")
                documents.append(f"Issue #{issue.number}: {issue.title}\n{issue.body}")
                issue_count += 1
                # Incrementally update cache
                with open(cache_file, 'w') as f:
                    json.dump(documents, f)
                print(f"Added issue #{issue.number} to documents and updated cache")
            print(f"Finished processing issues. Processed {issue_count} issues.")
        except GithubException as e:
            print(f"Error fetching issues: {str(e)}")
        
        # Fetch pull requests
        print("Fetching pull requests...")
        try:
            pr_count = 0
            for pr in repo.get_pulls(state="all"):
                print(f"Processing PR #{pr.number}: {pr.title}")
                documents.append(f"PR #{pr.number}: {pr.title}\n{pr.body}")
                pr_count += 1
                # Incrementally update cache
                with open(cache_file, 'w') as f:
                    json.dump(documents, f)
                print(f"Added PR #{pr.number} to documents and updated cache")
            print(f"Finished processing pull requests. Processed {pr_count} pull requests.")
        except GithubException as e:
            print(f"Error fetching pull requests: {str(e)}")
        
        # Fetch discussions (if available)
        print("Fetching discussions...")
        try:
            if repo.has_discussions:
                discussion_count = 0
                for discussion in repo.get_discussions():
                    print(f"Processing discussion: {discussion.title}")
                    documents.append(f"Discussion: {discussion.title}\n{discussion.body}")
                    discussion_count += 1
                    # Incrementally update cache
                    with open(cache_file, 'w') as f:
                        json.dump(documents, f)
                    print(f"Added discussion '{discussion.title}' to documents and updated cache")
                print(f"Finished processing discussions. Processed {discussion_count} discussions.")
            else:
                print("This repository does not have discussions enabled.")
        except GithubException as e:
            print(f"Error fetching discussions: {str(e)}")
    
    except GithubException as e:
        print(f"Error accessing repository: {str(e)}")
    
    print(f"Finished fetching data for {repo_name}")
    print(f"Total documents collected: {len(documents)}")
    return documents