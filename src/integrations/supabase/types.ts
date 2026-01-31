export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.1"
  }
  public: {
    Tables: {
      community_health_data: {
        Row: {
          avg_health_score: number | null
          common_symptoms: string[] | null
          created_at: string
          date: string
          id: string
          region: string
          risk_distribution: Json | null
          total_users: number | null
        }
        Insert: {
          avg_health_score?: number | null
          common_symptoms?: string[] | null
          created_at?: string
          date?: string
          id?: string
          region: string
          risk_distribution?: Json | null
          total_users?: number | null
        }
        Update: {
          avg_health_score?: number | null
          common_symptoms?: string[] | null
          created_at?: string
          date?: string
          id?: string
          region?: string
          risk_distribution?: Json | null
          total_users?: number | null
        }
        Relationships: []
      }
      health_goals: {
        Row: {
          created_at: string
          current_value: number | null
          end_date: string | null
          goal_name: string
          goal_type: string
          id: string
          is_active: boolean | null
          start_date: string
          target_value: number
          unit: string
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          current_value?: number | null
          end_date?: string | null
          goal_name: string
          goal_type: string
          id?: string
          is_active?: boolean | null
          start_date?: string
          target_value: number
          unit: string
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          current_value?: number | null
          end_date?: string | null
          goal_name?: string
          goal_type?: string
          id?: string
          is_active?: boolean | null
          start_date?: string
          target_value?: number
          unit?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      health_predictions: {
        Row: {
          created_at: string
          health_score: number | null
          id: string
          input_data: Json | null
          prediction_result: Json | null
          prediction_type: string
          user_id: string
        }
        Insert: {
          created_at?: string
          health_score?: number | null
          id?: string
          input_data?: Json | null
          prediction_result?: Json | null
          prediction_type: string
          user_id: string
        }
        Update: {
          created_at?: string
          health_score?: number | null
          id?: string
          input_data?: Json | null
          prediction_result?: Json | null
          prediction_type?: string
          user_id?: string
        }
        Relationships: []
      }
      health_summaries: {
        Row: {
          created_at: string
          email_sent: boolean | null
          health_score: number | null
          id: string
          period_end: string
          period_start: string
          summary_data: Json | null
          summary_type: string
          user_id: string
        }
        Insert: {
          created_at?: string
          email_sent?: boolean | null
          health_score?: number | null
          id?: string
          period_end: string
          period_start: string
          summary_data?: Json | null
          summary_type: string
          user_id: string
        }
        Update: {
          created_at?: string
          email_sent?: boolean | null
          health_score?: number | null
          id?: string
          period_end?: string
          period_start?: string
          summary_data?: Json | null
          summary_type?: string
          user_id?: string
        }
        Relationships: []
      }
      lifestyle_logs: {
        Row: {
          alcohol_units: number | null
          created_at: string
          daily_steps: number | null
          diet_quality: number | null
          exercise_minutes: number | null
          id: string
          log_date: string
          notes: string | null
          sleep_hours: number | null
          smoking: boolean | null
          stress_level: number | null
          user_id: string
          water_glasses: number | null
        }
        Insert: {
          alcohol_units?: number | null
          created_at?: string
          daily_steps?: number | null
          diet_quality?: number | null
          exercise_minutes?: number | null
          id?: string
          log_date?: string
          notes?: string | null
          sleep_hours?: number | null
          smoking?: boolean | null
          stress_level?: number | null
          user_id: string
          water_glasses?: number | null
        }
        Update: {
          alcohol_units?: number | null
          created_at?: string
          daily_steps?: number | null
          diet_quality?: number | null
          exercise_minutes?: number | null
          id?: string
          log_date?: string
          notes?: string | null
          sleep_hours?: number | null
          smoking?: boolean | null
          stress_level?: number | null
          user_id?: string
          water_glasses?: number | null
        }
        Relationships: []
      }
      notification_preferences: {
        Row: {
          created_at: string
          email_health_alerts: boolean
          email_monthly_summary: boolean
          email_weekly_summary: boolean
          id: string
          push_enabled: boolean
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          email_health_alerts?: boolean
          email_monthly_summary?: boolean
          email_weekly_summary?: boolean
          id?: string
          push_enabled?: boolean
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          email_health_alerts?: boolean
          email_monthly_summary?: boolean
          email_weekly_summary?: boolean
          id?: string
          push_enabled?: boolean
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      profiles: {
        Row: {
          age: number | null
          created_at: string
          full_name: string | null
          gender: string | null
          height_cm: number | null
          id: string
          updated_at: string
          user_id: string
          weight_kg: number | null
        }
        Insert: {
          age?: number | null
          created_at?: string
          full_name?: string | null
          gender?: string | null
          height_cm?: number | null
          id?: string
          updated_at?: string
          user_id: string
          weight_kg?: number | null
        }
        Update: {
          age?: number | null
          created_at?: string
          full_name?: string | null
          gender?: string | null
          height_cm?: number | null
          id?: string
          updated_at?: string
          user_id?: string
          weight_kg?: number | null
        }
        Relationships: []
      }
      symptom_checks: {
        Row: {
          ai_analysis: string | null
          created_at: string
          id: string
          recommendations: string[] | null
          risk_level: string | null
          risk_score: number | null
          symptoms: string[]
          user_id: string
        }
        Insert: {
          ai_analysis?: string | null
          created_at?: string
          id?: string
          recommendations?: string[] | null
          risk_level?: string | null
          risk_score?: number | null
          symptoms: string[]
          user_id: string
        }
        Update: {
          ai_analysis?: string | null
          created_at?: string
          id?: string
          recommendations?: string[] | null
          risk_level?: string | null
          risk_score?: number | null
          symptoms?: string[]
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
