# src/preprocessing.py
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class TextPreprocessor:
    """Handles all text cleaning and preprocessing"""
    
    @staticmethod
    def clean_text(text):
        """
        Clean resume text by removing URLs, special characters, etc.
        Matches synopsis requirements for preprocessing
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub('http\\S+\\s*', ' ', text)
        
        # Remove RT and cc
        text = re.sub('RT|cc', ' ', text)
        
        # Remove hashtags
        text = re.sub('#\\S+', '', text)
        
        # Remove mentions
        text = re.sub('@\\S+', ' ', text)
        
        # Remove punctuation (tokenization step)
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
        
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7f]', r' ', text)
        
        # Remove extra spaces
        text = re.sub('\s+', ' ', text)
        
        # Lowercasing (as mentioned in synopsis)
        text = text.lower()
        
        return text.strip()
    
    @staticmethod
    def extract_skills(text, skill_set):
        """
        Extract skills from text using a predefined skill dictionary
        """
        if not isinstance(text, str):
            return set()
        
        text_lower = text.lower()
        found_skills = set()
        
        for skill in skill_set:
            # Look for whole word matches
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        
        return found_skills
    
    @staticmethod
    def extract_experience(text):
        """
        Extract years of experience from text
        Used for experience relevance scoring
        """
        if not isinstance(text, str):
            return 0
        
        text_lower = text.lower()
        
        # Patterns for experience extraction
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s+of\s+experience',
            r'experience\s*:?\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)[-\s]to[-\s](\d+)\s*(?:years?|yrs?)\s+experience',
            r'(\d+)\+?\s*(?:years?|yrs?)'
        ]
        
        years_found = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle range like "5-7 years" - take average
                    years_found.append((int(match[0]) + int(match[1])) / 2)
                else:
                    years_found.append(int(match))
        
        if years_found:
            return max(years_found)  # Return highest mentioned
        
        return 0


class SkillDatabase:
    """Manages the skills database - expandable based on dataset"""
    
    def __init__(self):
        self.skills = self._create_skills_database()
    
    def _create_skills_database(self):
        """Create comprehensive skills database based on common resume categories"""
        skills = {
            # Programming Languages
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 
            'kotlin', 'typescript', 'go', 'rust', 'scala', 'r', 'matlab', 'sql',
            'html', 'css', 'bash', 'shell', 'perl',
            
            # Data Science & ML
            'machine learning', 'deep learning', 'data science', 'data analysis',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'matplotlib', 'seaborn', 'plotly', 'nlp', 'computer vision', 'opencv',
            'tableau', 'power bi', 'excel', 'statistics', 'regression',
            'classification', 'clustering', 'neural networks',
            
            # Web Development
            'react', 'angular', 'vue', 'node.js', 'nodejs', 'django', 'flask',
            'spring boot', 'bootstrap', 'jquery', 'rest api', 'graphql',
            'express', 'next.js', 'html5', 'css3', 'sass', 'less',
            
            # Cloud & DevOps
            'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes',
            'jenkins', 'git', 'github', 'gitlab', 'ci/cd', 'terraform',
            'ansible', 'puppet', 'chef', 'linux', 'unix', 'devops',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server',
            'redis', 'elasticsearch', 'cassandra', 'dynamodb', 'firebase',
            'nosql', 'database',
            
            # Soft Skills
            'project management', 'agile', 'scrum', 'leadership',
            'communication', 'teamwork', 'problem solving', 'critical thinking',
            'time management', 'presentation', 'negotiation', 'mentoring',
            
            # HR & Management
            'recruitment', 'hiring', 'onboarding', 'employee relations',
            'hr policies', 'performance management', 'training', 'hr',
            'talent acquisition', 'payroll', 'benefits',
            
            # Testing
            'testing', 'qa', 'quality assurance', 'selenium', 'junit', 
            'testng', 'cucumber', 'postman', 'jmeter', 'manual testing', 
            'automation testing', 'unit testing',
            
            # Java specific (from dataset preview)
            'j2ee', 'ejb', 'jsp', 'servlets', 'hibernate', 'struts',
            
            # Other technical
            'excel', 'word', 'powerpoint', 'outlook', 'sharepoint'
        }
        
        return skills
    
    def save(self, filepath):
        """Save skills database to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.skills, f)
    
    def load(self, filepath):
        """Load skills database from file"""
        with open(filepath, 'rb') as f:
            self.skills = pickle.load(f)
        return self.skills