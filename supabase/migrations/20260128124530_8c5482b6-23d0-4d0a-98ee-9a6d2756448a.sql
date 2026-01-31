-- Create health goals table
CREATE TABLE public.health_goals (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  goal_type TEXT NOT NULL, -- 'steps', 'sleep', 'water', 'exercise', 'weight', 'custom'
  goal_name TEXT NOT NULL,
  target_value NUMERIC NOT NULL,
  unit TEXT NOT NULL, -- 'steps', 'hours', 'glasses', 'minutes', 'kg', etc.
  current_value NUMERIC DEFAULT 0,
  start_date DATE NOT NULL DEFAULT CURRENT_DATE,
  end_date DATE,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.health_goals ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can view their own health goals"
ON public.health_goals
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own health goals"
ON public.health_goals
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own health goals"
ON public.health_goals
FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own health goals"
ON public.health_goals
FOR DELETE
USING (auth.uid() = user_id);

-- Create trigger for automatic timestamp updates
CREATE TRIGGER update_health_goals_updated_at
BEFORE UPDATE ON public.health_goals
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();