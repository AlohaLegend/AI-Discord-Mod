import discord
from discord.ext import commands, tasks
from discord.ext.commands import has_permissions, MissingPermissions
from ai_discord_functions import message_is_safe
from discord import app_commands
import os
from dotenv import load_dotenv
import asyncio
import json

load_dotenv()

# Create a lock for each file
servers_lock = asyncio.Lock()
warnings_lock = asyncio.Lock()
sensitivity_lock = asyncio.Lock()

# Save servers settings to file
async def save_servers():
    async with servers_lock:
        try:
            with open("servers.json", "w") as file:
                json.dump(servers, file)
        except IOError as e:
            print(f"Error saving servers: {e}")

# Save warnings to file
async def save_warnings():
    async with warnings_lock:
        try:
            with open("warnings.json", "w") as file:
                json.dump(warning_list, file)
        except IOError as e:
            print(f"Error saving warnings: {e}")


# Load servers settings from file
try:
    with open("servers.json", "r") as file:
        servers = json.load(file)
except FileNotFoundError:
    servers = {}

try:
    with open("warnings.json", "r") as file:
        warning_list = json.load(file)
except FileNotFoundError:
    warning_list = {}

async def save_sensitivity():
    async with sensitivity_lock:
        try:
            with open("sensitivity.json", "w") as file:
                json.dump(sensitivity, file)
        except IOError as e:
            print(f"Error saving sensitivity: {e}")

# Load sensitivity settings from file
try:
    with open("sensitivity.json", "r") as file:
        sensitivity = json.load(file)
except FileNotFoundError:
    sensitivity = {}


intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.tree.command(name="help", description="Shows commands and information for the Stanley AI bot.")
@app_commands.guild_only()
@app_commands.default_permissions(administrator=True)
async def aihelp(interaction: discord.Interaction):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return

    await interaction.response.send_message(
        """
**Help:**
```
help: Shows this information.
set_warnings <warnings>: Sets the number of warnings a user can have before muting them. Set to 0 to mute immediately on first offense.
set_mute_time <time>: Sets the amount of time a user is muted after reaching the warning limit. Example: 1d, 3m, 5s, 6h
use_warnings <boolean>: Toggle whether Stanley uses a warning system before muting or acts immediately.
set_sensitivity <float from 0-1>: Adjusts the image moderation strictness. 0 = lenient, 1 = strict.
set_threshold <category> <threshold>: Set category-specific moderation thresholds. Categories: harassment, hate, violence, sexual, self_harm
show_thresholds: Displays the current thresholds for all moderation categories.
set_logs_channel <channel id>: Sets the channel where Stanley logs flagged messages. Stanley needs permission to send messages there.
delete_flagged_messages <true|false>: If true, flagged messages will be deleted. If false, Stanley will react with 🚫 instead.
```

Note the default presets:
```
set_warnings: 3
set_mute_time: 10m
use_warnings: False
set_sensitivity: 0.5
set_logs_channel: None (will not log any deletions)
```

""", ephemeral = True)

@bot.tree.command(name="set_logs_channel", description="Set a server wide channel id for logging messages.")
@app_commands.guild_only()
@app_commands.default_permissions(administrator=True)
@app_commands.describe(logs_channel_id = "Logs Channel ID")
async def set_logs_channel(interaction: discord.Interaction, logs_channel_id: str):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(f"You do not have permission to use this command.", ephemeral=True)
        return
    try:
        servers[str(interaction.guild.id)] = servers.get(str(interaction.guild.id), {})
        servers[str(interaction.guild.id)]['logs_channel_id'] = logs_channel_id[2:-1]
        await save_servers()
        await interaction.response.send_message(f"**Successfully set logs channel id to: {logs_channel_id}**", ephemeral=True)
    except:
        await interaction.response.send_message("**Failed to parse logs channel id. Logs Channel ID must be an integer.**", ephemeral=True)

@bot.tree.command(name="use_warnings", description="Whether to automatically mute users after a certain amount of warnings.")
@app_commands.guild_only()
@app_commands.default_permissions(administrator=True)
@app_commands.describe(use_warnings = "Use Warnings")
async def use_warnings(interaction: discord.Interaction, use_warnings: bool):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(f"You do not have permission to use this command.", ephemeral=True)
        return
    servers[str(interaction.guild.id)] = servers.get(str(interaction.guild.id), {})
    servers[str(interaction.guild.id)]['use_warnings'] = use_warnings
    await save_servers()
    await interaction.response.send_message(f"Successfully set use_warnings to **{use_warnings}**.", ephemeral=True)

# @bot.tree.command(name="set_sensitivity", description="Set a server wide image moderation sensitivity.")
# @app_commands.guild_only()
# @app_commands.default_permissions(administrator=True)
# @app_commands.describe(sensitivity = "Image Moderation Sensitivity")
# async def set_sensitivity(interaction: discord.Interaction, sensitivity: float):
#     if not interaction.user.guild_permissions.administrator:
#         await interaction.response.send_message(f"You do not have permission to use this command.", ephemeral=True)
#         return
#     if sensitivity > 1:
#         await interaction.response.send_message("**Failed to parse sensitivity. Sensitivity must be a number from 0-1.**", ephemeral=True)
#         return
#     try:
#         servers[str(interaction.guild.id)] = servers.get(str(interaction.guild.id), {})
#         servers[str(interaction.guild.id)]['sensitivity'] = sensitivity
#         await save_servers()
#         await interaction.response.send_message(f"**Successfully set image moderation sensitivity to: {sensitivity}**", ephemeral=True)
#     except:
#         await interaction.response.send_message("**Failed to parse sensitivity. Sensitivity must be a number from 0-1.**", ephemeral=True)

@bot.tree.command(name="set_warnings", description="Set a server wide warnings limit before muting a member.")
@app_commands.guild_only()
@app_commands.default_permissions(administrator=True)
@app_commands.describe(warning_count = "Warning Count")
async def set_warnings(interaction: discord.Interaction, warning_count: int):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(f"You do not have permission to use this command.", ephemeral=True)
        return
    try:
        warnings = warning_count
        servers[str(interaction.guild.id)] = servers.get(str(interaction.guild.id), {})
        servers[str(interaction.guild.id)]['warnings'] = warnings
        await save_servers()
        await interaction.response.send_message(f"**Successfully set warnings to: {warnings}**", ephemeral=True)
    except:
        await interaction.response.send_message("**Failed to parse warnings. Warnings must be an integer.**", ephemeral=True)

@bot.tree.command(name="set_mute_time", description="Set a server wide mute time to mute a member for.")
@app_commands.guild_only()
@app_commands.default_permissions(administrator=True)
@app_commands.describe(mute_time = "Mute Time")
async def set_mute_time(interaction: discord.Interaction, mute_time: str):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(f"You do not have permission to use this command.", ephemeral=True)
        return
    try:
        servers[str(interaction.guild.id)] = servers.get(str(interaction.guild.id), {})
        servers[str(interaction.guild.id)]['mute_time'] = mute_time
        await save_servers()
        await interaction.response.send_message(f"**Successfully set mute time to {mute_time}**", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message("**Invalid duration input**", ephemeral=True)

@bot.tree.command(name="delete_flagged_messages", description="Enable or disable deletion of flagged messages.")
@app_commands.guild_only()
@app_commands.default_permissions(administrator=True)
@app_commands.describe(enabled="Whether flagged messages should be deleted")
async def delete_flagged_messages(interaction: discord.Interaction, enabled: bool):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return
    servers[str(interaction.guild.id)] = servers.get(str(interaction.guild.id), {})
    servers[str(interaction.guild.id)]['delete_flagged_messages'] = enabled
    await save_servers()
    await interaction.response.send_message(
        f"Flagged messages will now be {'deleted' if enabled else 'preserved with a 🚫 reaction'}.", ephemeral=True)

@bot.tree.command(name="set_threshold", description="Set moderation threshold for a specific category.")
@app_commands.guild_only()
@app_commands.default_permissions(administrator=True)
@app_commands.describe(category="Moderation category", threshold="Threshold from 0.0 to 1.0")
async def set_threshold(interaction: discord.Interaction, category: str, threshold: float):
    valid_categories = ["harassment", "hate", "violence", "sexual", "self_harm"]
    category = category.lower()

    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return

    if category not in valid_categories:
        await interaction.response.send_message(f"Invalid category. Choose from: {', '.join(valid_categories)}", ephemeral=True)
        return

    if not (0.0 <= threshold <= 1.0):
        await interaction.response.send_message("Threshold must be between 0.0 and 1.0", ephemeral=True)
        return

    server_id = str(interaction.guild.id)
    servers.setdefault(server_id, {})
    servers[server_id].setdefault("moderation_thresholds", {})
    servers[server_id]["moderation_thresholds"][category] = threshold

    await interaction.response.send_message(f"Set **{category}** threshold to **{threshold}**", ephemeral=True)

@bot.tree.command(name="show_thresholds", description="Displays all moderation thresholds set for this server.")
@app_commands.guild_only()
@app_commands.default_permissions(administrator=True)
async def show_thresholds(interaction: discord.Interaction):
    server_id = str(interaction.guild.id)
    thresholds = servers.get(server_id, {}).get("moderation_thresholds", {})

    if not thresholds:
        await interaction.response.send_message("No custom moderation thresholds have been set for this server.", ephemeral=True)
        return

    formatted = "\n".join([f"**{cat}**: {thresh}" for cat, thresh in thresholds.items()])
    await interaction.response.send_message(
        f"**Current Moderation Thresholds:**\n{formatted}",
        ephemeral=True
    )

@bot.tree.command(name="stanley", description="Say hi with Stanley!")
async def stanley(interaction: discord.Interaction):
    # Replace 'your_emoji_name' and the ID with your actual emoji info
    await interaction.response.send_message(f"Hi {interaction.user.mention}! Stanley says Hi! <:stanley:1221959011226746950>")

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_TRIGGERING_WORDS = os.getenv("USE_TRIGGERING_WORDS")

if USE_TRIGGERING_WORDS == "True":
    TRIGGERING_WORDS_FILE = os.getenv("TRIGGERING_WORDS")
    if TRIGGERING_WORDS_FILE:
        with open(TRIGGERING_WORDS_FILE, "r") as file:
            TRIGGERING_WORDS = file.read().split(",")
    else:
        TRIGGERING_WORDS = []
else:
    TRIGGERING_WORDS = []

if not BOT_TOKEN or not OPENAI_API_KEY:
    print("You did not set your .env file correctly.")
    exit()

from datetime import timedelta

async def tempmute(interaction_or_channel, member: discord.Member):
    guild = member.guild
    warnings = servers[str(guild.id)].get('warnings', 3)
    time_str = servers[str(guild.id)].get('mute_time', '10m')
    reason = f"Exceeded {warnings} inappropriate messages."

    try:
        duration_num = int(time_str[:-1])
        unit = time_str[-1]
        if unit == 's':
            timeout_duration = timedelta(seconds=duration_num)
        elif unit == 'm':
            timeout_duration = timedelta(minutes=duration_num)
        elif unit == 'h':
            timeout_duration = timedelta(hours=duration_num)
        elif unit == 'd':
            timeout_duration = timedelta(days=duration_num)
        else:
            raise ValueError("Invalid time unit")
    except Exception as e:
        await interaction_or_channel.send("Invalid mute duration format. Use something like `10m`, `2h`, etc.")
        return

    try:
        await member.timeout(timeout_duration, reason=reason)
        await interaction_or_channel.send(
            embed=discord.Embed(
                title="User Timed Out",
                description=f"{member.mention} was timed out for {time_str} due to inappropriate message.",
                color=discord.Color.orange()
            )
        )

        # ✅ Log to logs channel (if configured)
        logs_channel_id = servers[str(guild.id)].get('logs_channel_id')
        if logs_channel_id:
            logs_channel = bot.get_channel(int(logs_channel_id))
            if logs_channel:
                await logs_channel.send(
                    embed=discord.Embed(
                        title="Timeout Logged",
                        description=f"{member.mention} was timed out for {time_str}.\nReason: {reason}",
                        color=discord.Color.dark_orange()
                    )
                )

    except discord.Forbidden:
        await interaction_or_channel.send("I do not have permission to timeout this user.")
    except Exception as e:
        await interaction_or_channel.send(f"Failed to timeout: {e}")

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)

@bot.event
async def on_message(message):
    await bot.wait_until_ready()
    if message.author.id == bot.user.id:
        return

    sent_message = message
    guild = message.guild

    if str(guild.id) not in servers:
        servers[str(guild.id)] = {'use_warnings': False, 'warnings': 3, 'mute_time': '10m'}
        await save_servers()

    use_warnings = servers[str(guild.id)].get('use_warnings', False)
    warnings = servers[str(guild.id)].get('warnings', 3)

    if str(guild.id) not in warning_list:
        warning_list[str(guild.id)] = {}
        await save_warnings()

    # ✅ OpenAI Moderation Check with updated threshold logic
    is_safe, flagged_category, flagged_score, flagged_threshold = await message_is_safe(
        message.content, OPENAI_API_KEY, servers, guild.id
    )

    if not is_safe:
        try:
            delete_flagged = servers[str(guild.id)].get('delete_flagged_messages', False)

            if delete_flagged:
                await message.delete()
            else:
                await message.add_reaction("🚫")

            logs_channel_id = servers[str(guild.id)].get('logs_channel_id')
            if logs_channel_id:
                logs_channel = bot.get_channel(int(logs_channel_id))
                if logs_channel:
                    await logs_channel.send(
                        f"Reacted to {sent_message.author.mention}'s message because it was inappropriate.\n"
                        f"Message content: '{sent_message.content}'\n"
                        f"Flagged category: **{flagged_category}**, "
                        f"Score: **{flagged_score:.2f}**, Threshold: **{flagged_threshold:.2f}**"
                    )

            if not use_warnings:
                await sent_message.channel.send(
                    f"{sent_message.author.mention}, your message was flagged as inappropriate. 🚫"
                )
                return
                
            if use_warnings and warnings == 0:
                await sent_message.channel.send(
                    f"{sent_message.author.mention}, your message was flagged as inappropriate. "
                    f"You are being muted immediately. 🚫"
                )
                await tempmute(sent_message.channel, sent_message.author)
                return

            # Handle Warnings
            if message.author.id in warning_list[str(guild.id)]:
                warning_list[str(guild.id)][message.author.id] += 1
                await save_warnings()

                if warning_list[str(guild.id)][message.author.id] >= warnings:
                    await sent_message.channel.send(
                        f"{sent_message.author.mention}, your message was flagged as inappropriate. "
                        f"You have reached the warning limit. Muting you now. 🚫"
                    )
                    await tempmute(sent_message.channel, sent_message.author)
                    warning_list[str(guild.id)][message.author.id] = 0
                    await save_warnings()
                else:
                    remaining = warnings - warning_list[str(guild.id)][message.author.id]
                    await sent_message.channel.send(
                        f"{sent_message.author.mention}, your message was flagged as inappropriate. 🚫 "
                        f"You have {remaining} warnings left."
                    )
            else:
                warning_list[str(guild.id)][message.author.id] = 1
                await save_warnings()
                remaining = warnings - 1
                await sent_message.channel.send(
                    f"{sent_message.author.mention}, your message was flagged as inappropriate. 🚫 "
                    f"You have {remaining} warnings left."
                )

        except Exception as e:
            print(f"Error handling inappropriate message: {e}")

    await bot.process_commands(message)

@bot.event
async def on_message_edit(before, after):
    if after.author.bot:
        return
    if after.content.startswith(bot.command_prefix):
        return
    await on_message(after)
    
bot.run(BOT_TOKEN)
