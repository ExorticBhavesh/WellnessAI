-- Create profiles table for user health data
CREATE TABLE public.profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    full_name TEXT,
    age INTEGER,
    gender TEXT CHECK (gender IN ('male', 'female', 'other', 'prefer_not_to_say')),
    height_cm NUMERIC(5,2),
    weight_kg NUMERIC(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

-- Enable RLS on profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view their own profile"
ON public.profiles FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own profile"
ON public.profiles FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own profile"
ON public.profiles FOR UPDATE
USING (auth.uid() = user_id);

-- Create lifestyle_logs table
CREATE TABLE public.lifestyle_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    log_date DATE NOT NULL DEFAULT CURRENT_DATE,
    sleep_hours NUMERIC(4,2),
    exercise_minutes INTEGER,
    daily_steps INTEGER,
    diet_quality INTEGER CHECK (diet_quality >= 1 AND diet_quality <= 10),
    stress_level INTEGER CHECK (stress_level >= 1 AND stress_level <= 10),
    water_glasses INTEGER,
    smoking BOOLEAN DEFAULT false,
    alcohol_units INTEGER DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    UNIQUE(user_id, log_date)
);

-- Enable RLS on lifestyle_logs
ALTER TABLE public.lifestyle_logs ENABLE ROW LEVEL SECURITY;

-- Lifestyle logs policies
CREATE POLICY "Users can view their own logs"
ON public.lifestyle_logs FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own logs"
ON public.lifestyle_logs FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own logs"
ON public.lifestyle_logs FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own logs"
ON public.lifestyle_logs FOR DELETE
USING (auth.uid() = user_id);

-- Create symptoms table for symptom checker
CREATE TABLE public.symptom_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    symptoms TEXT[] NOT NULL,
    risk_level TEXT CHECK (risk_level IN ('low', 'medium', 'high')),
    risk_score NUMERIC(5,2),
    ai_analysis TEXT,
    recommendations TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

-- Enable RLS on symptom_checks
ALTER TABLE public.symptom_checks ENABLE ROW LEVEL SECURITY;

-- Symptom checks policies
CREATE POLICY "Users can view their own symptom checks"
ON public.symptom_checks FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own symptom checks"
ON public.symptom_checks FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- Create health_predictions table
CREATE TABLE public.health_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    prediction_type TEXT NOT NULL,
    input_data JSONB,
    prediction_result JSONB,
    health_score NUMERIC(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

-- Enable RLS on health_predictions
ALTER TABLE public.health_predictions ENABLE ROW LEVEL SECURITY;

-- Health predictions policies
CREATE POLICY "Users can view their own predictions"
ON public.health_predictions FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own predictions"
ON public.health_predictions FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- Create community_health_data for anonymized aggregated data
CREATE TABLE public.community_health_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region TEXT NOT NULL,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    avg_health_score NUMERIC(5,2),
    common_symptoms TEXT[],
    total_users INTEGER DEFAULT 0,
    risk_distribution JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

-- Enable RLS on community_health_data (readable by all authenticated users)
ALTER TABLE public.community_health_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Authenticated users can view community data"
ON public.community_health_data FOR SELECT
TO authenticated
USING (true);

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = public;

-- Add trigger for profiles updated_at
CREATE TRIGGER update_profiles_updated_at
BEFORE UPDATE ON public.profiles
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- Create function to auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (user_id)
    VALUES (NEW.id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

-- Create trigger for new user signup
CREATE TRIGGER on_auth_user_created
AFTER INSERT ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_new_user();

-- Insert sample community health data
INSERT INTO public.community_health_data (region, date, avg_health_score, common_symptoms, total_users, risk_distribution) VALUES
('North Region', CURRENT_DATE - INTERVAL '30 days', 72.5, ARRAY['fatigue', 'headache'], 1250, '{"low": 65, "medium": 28, "high": 7}'),
('North Region', CURRENT_DATE - INTERVAL '23 days', 71.8, ARRAY['fatigue', 'cough'], 1280, '{"low": 63, "medium": 30, "high": 7}'),
('North Region', CURRENT_DATE - INTERVAL '16 days', 73.2, ARRAY['fatigue', 'headache'], 1320, '{"low": 66, "medium": 27, "high": 7}'),
('North Region', CURRENT_DATE - INTERVAL '9 days', 74.1, ARRAY['headache', 'stress'], 1350, '{"low": 68, "medium": 26, "high": 6}'),
('North Region', CURRENT_DATE - INTERVAL '2 days', 75.0, ARRAY['stress', 'fatigue'], 1380, '{"low": 70, "medium": 25, "high": 5}'),
('South Region', CURRENT_DATE - INTERVAL '30 days', 68.3, ARRAY['allergies', 'fatigue'], 980, '{"low": 55, "medium": 35, "high": 10}'),
('South Region', CURRENT_DATE - INTERVAL '23 days', 69.1, ARRAY['allergies', 'headache'], 1020, '{"low": 57, "medium": 34, "high": 9}'),
('South Region', CURRENT_DATE - INTERVAL '16 days', 70.5, ARRAY['fatigue', 'allergies'], 1050, '{"low": 60, "medium": 32, "high": 8}'),
('South Region', CURRENT_DATE - INTERVAL '9 days', 71.2, ARRAY['stress', 'fatigue'], 1080, '{"low": 62, "medium": 30, "high": 8}'),
('South Region', CURRENT_DATE - INTERVAL '2 days', 72.0, ARRAY['fatigue', 'stress'], 1100, '{"low": 64, "medium": 29, "high": 7}'),
('East Region', CURRENT_DATE - INTERVAL '30 days', 76.8, ARRAY['stress', 'insomnia'], 1450, '{"low": 72, "medium": 23, "high": 5}'),
('East Region', CURRENT_DATE - INTERVAL '23 days', 77.2, ARRAY['stress', 'headache'], 1480, '{"low": 73, "medium": 22, "high": 5}'),
('East Region', CURRENT_DATE - INTERVAL '16 days', 77.8, ARRAY['insomnia', 'stress'], 1510, '{"low": 74, "medium": 22, "high": 4}'),
('East Region', CURRENT_DATE - INTERVAL '9 days', 78.5, ARRAY['stress', 'fatigue'], 1540, '{"low": 76, "medium": 20, "high": 4}'),
('East Region', CURRENT_DATE - INTERVAL '2 days', 79.2, ARRAY['fatigue', 'stress'], 1570, '{"low": 78, "medium": 19, "high": 3}'),
('West Region', CURRENT_DATE - INTERVAL '30 days', 70.2, ARRAY['cough', 'fatigue'], 890, '{"low": 58, "medium": 32, "high": 10}'),
('West Region', CURRENT_DATE - INTERVAL '23 days', 69.8, ARRAY['cough', 'fever'], 920, '{"low": 56, "medium": 33, "high": 11}'),
('West Region', CURRENT_DATE - INTERVAL '16 days', 71.5, ARRAY['fatigue', 'cough'], 950, '{"low": 60, "medium": 31, "high": 9}'),
('West Region', CURRENT_DATE - INTERVAL '9 days', 72.8, ARRAY['fatigue', 'headache'], 980, '{"low": 63, "medium": 29, "high": 8}'),
('West Region', CURRENT_DATE - INTERVAL '2 days', 73.5, ARRAY['headache', 'stress'], 1010, '{"low": 65, "medium": 28, "high": 7}');